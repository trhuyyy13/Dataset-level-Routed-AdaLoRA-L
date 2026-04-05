import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class APILayerImportance:
    def __init__(self, model_name_or_path, imp_func='fisher'):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if imp_func == 'fisher':
            self.imp_func = self._compute_fisher_information
        else:
            self.imp_func = self._compute_sensitivity
        
        self.edit_module_names = []
        if 'starcoder' in model_name_or_path.lower():
            for i in range(len(self.model.model.layers)):
                self.edit_module_names.append(f"model.layers.{i}.self_attn.q_proj")
                self.edit_module_names.append(f"model.layers.{i}.self_attn.v_proj")
        else:
            proj_name = 'down_proj'
            for i in range(len(self.model.model.layers)):
                self.edit_module_names.append(f"model.layers.{i}.mlp.{proj_name}")
        
        self.module_importance = {name: 0.0 for name in self.edit_module_names}
        self._freeze_non_edit_params()
        
    def _freeze_non_edit_params(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for module_name in self.edit_module_names:
            try:
                module = self.model.get_submodule(module_name)
                for param in module.parameters():
                    param.requires_grad = True
            except AttributeError:
                print(f"Warning: Module {module_name} does not exist, skipping.")
    
    def compute_layer_importance(self, input_text, target_api):
        scores = {name: 0.0 for name in self.edit_module_names}
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        offsets = inputs["offset_mapping"][0].cpu().numpy()
        original_text = input_text

        api_start = original_text.find(target_api)
        if api_start == -1:
            raise ValueError(f"The target API was not found in the input text: {target_api}")
        api_end = api_start + len(target_api)

        api_token_indices = []
        for token_idx, (start, end) in enumerate(offsets):
            if not (end <= api_start or start >= api_end):
                api_token_indices.append(token_idx)

        if not api_token_indices:
            raise ValueError(f"No token corresponding to the target API was found: {target_api}")
        
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

        mask = torch.zeros_like(token_losses, device=self.model.device)
        for idx in api_token_indices:
            if 0 <= idx - 1 < mask.size(1):
                mask[0, idx - 1] = 1.0

        focused_loss = (token_losses * mask).sum()
        focused_loss.backward()

        for module_name in self.edit_module_names:
            try:
                module = self.model.get_submodule(module_name)
                total_importance = 0.0
                param_count = 0
                
                for param in module.parameters():
                    if param.grad is not None:
                        total_importance += self.imp_func(param, param.grad).item()
                        param_count += 1
                
                if param_count > 0:
                    scores[module_name] = total_importance / param_count
            except AttributeError:
                continue
                
        for module_name in self.edit_module_names:
            if scores[module_name]:
                self.module_importance[module_name] = scores[module_name]

        layer_scores = {}
        for name, score in self.module_importance.items():
            layer_idx = int(name.split(".")[2])
            if layer_idx not in layer_scores:
                layer_scores[layer_idx] = 0.0
            layer_scores[layer_idx] += score

        sorted_result = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_result
    
    def _compute_sensitivity(self, param, grad):
        if grad is None:
            return torch.tensor(0.0, device=param.device)
        return (param * grad).abs().mean()
    
    def _compute_fisher_information(self, param, grad):
        if grad is None:
            return torch.tensor(0.0, device=param.device)
        return (grad ** 2).mean()
    
    def initialize_importance(self):
        self.module_importance = {name: 0.0 for name in self.edit_module_names}