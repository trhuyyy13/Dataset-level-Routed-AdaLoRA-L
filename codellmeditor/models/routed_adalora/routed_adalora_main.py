import re
import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from peft import get_peft_model, AdaLoraConfig, TaskType, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .routed_adalora_hparams import RoutedAdaLoRAHyperParams

LOG = logging.getLogger(__name__)


def sanitize_adapter_name(api_name: str) -> str:
    """Convert API name to valid PEFT adapter name (alphanumeric + underscore only)."""
    return re.sub(r'[^a-zA-Z0-9]', '_', api_name)


def apply_routed_adalora_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: RoutedAdaLoRAHyperParams,
        layer_config: Dict,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Dataset-level Routed AdaLoRA-L:
    1. Create one adapter per API on its specific layers
    2. Train all APIs in round-robin fashion
    3. Return edited model with all adapters
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    edited_model, adapter_map = execute_routed_adalora(
        model, tok, requests, hparams, layer_config
    )

    return edited_model, adapter_map


def execute_routed_adalora(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: RoutedAdaLoRAHyperParams,
        layer_config: Dict,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, str]]:
    """
    Core routing + training logic.
    
    Args:
        layer_config: {
            "common_layers": [int, ...],
            "api_specific_layers": {"api_name": [int, ...], ...}
        }
    """
    common_layers = set(layer_config.get("common_layers", []))
    api_specific_layers = layer_config.get("api_specific_layers", {})
    
    LOG.info(f"Common layers (frozen): {sorted(common_layers)}")
    LOG.info(f"Number of APIs with specific layers: {len(api_specific_layers)}")

    # ── Step 1: Group requests by API ──
    api_to_requests = defaultdict(list)
    for req in requests:
        api_name = req.get("target_api", "unknown")
        api_to_requests[api_name].append(req)
    
    LOG.info(f"Total requests: {len(requests)}, grouped into {len(api_to_requests)} APIs")
    
    # ── Step 2: Prepare model for training ──
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # ── Step 3: Create one adapter per API ──
    if hparams.lora_type == "adalora":
        ConfigClass = AdaLoraConfig
    elif hparams.lora_type == "lora":
        ConfigClass = LoraConfig
    else:
        raise NotImplementedError(f"lora_type {hparams.lora_type} not supported")
    
    adapter_map = {}  # api_name -> adapter_name
    first_adapter = True
    
    for api_name, api_requests in api_to_requests.items():
        adapter_name = sanitize_adapter_name(api_name)
        
        # Get layers for this API (exclude common layers)
        specific_layers = api_specific_layers.get(api_name, [])
        if not specific_layers:
            LOG.warning(f"No specific layers found for {api_name}, skipping adapter")
            continue
        
        specific_layers = [l for l in specific_layers if l not in common_layers]
        if not specific_layers:
            LOG.warning(f"All layers for {api_name} are common layers, skipping")
            continue
            
        adapter_config = ConfigClass(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha,
            lora_dropout=hparams.lora_dropout,
            layers_to_transform=specific_layers,
            target_modules=hparams.target_modules,
        )
        
        if first_adapter:
            model = get_peft_model(model, adapter_config, adapter_name=adapter_name)
            first_adapter = False
        else:
            model.add_adapter(adapter_name, adapter_config)
        
        adapter_map[api_name] = adapter_name
        LOG.info(f"  Adapter '{adapter_name}' -> layers {specific_layers} "
                 f"({len(api_requests)} samples)")
    
    if not adapter_map:
        LOG.error("No adapters were created! Check layer_config.")
        return model, adapter_map
    
    model.is_parallelizable = True
    model.model_parallel = True
    
    # Print trainable parameters summary
    model.print_trainable_parameters()
    
    # ── Step 4: Training loop with routing ──
    device = torch.device(f'cuda:{hparams.device}')
    
    opt = torch.optim.Adam(
        model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    
    loss_meter = AverageMeter()
    api_names = list(adapter_map.keys())
    
    for epoch in range(hparams.num_epochs):
        loss_meter.reset()
        epoch_loss = AverageMeter()
        
        # Shuffle API order each epoch
        import random
        random.shuffle(api_names)
        
        for api_name in api_names:
            adapter_name = adapter_map[api_name]
            api_requests = api_to_requests[api_name]
            
            # ── ROUTING: switch to this API's adapter ──
            model.set_adapter(adapter_name)
            
            # Process requests in batches
            for batch_reqs in chunks(api_requests, hparams.batch_size):
                mask_token = -100
                opt.zero_grad()
                
                texts = [r["prompt"] for r in batch_reqs]
                targets = [r["target_new"] for r in batch_reqs]
                
                full_prompt = [f"{p} {l}" for p, l in zip(texts, targets)]
                prompt_ids = tok(
                    list(texts), return_tensors="pt",
                    padding=True, truncation=True
                )["input_ids"]
                num_prompt_toks = [
                    int((i != tok.pad_token_id).sum()) for i in prompt_ids
                ]
                tokens = tok(
                    full_prompt, return_tensors="pt",
                    padding=True, truncation=True
                )
                bs = tokens["input_ids"].shape[0]
                tokens["labels"] = tokens["input_ids"].clone()
                num_pad_toks = [
                    int((i == tok.pad_token_id).sum()) for i in tokens["labels"]
                ]
                for j in range(len(texts)):
                    tokens["labels"][j][
                        num_pad_toks[j]:num_pad_toks[j]+num_prompt_toks[j]
                    ] = mask_token
                tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
                tokens = tokens.to(device)
                
                pred = model(**tokens)
                loss = pred.loss
                
                if loss is not None:
                    loss.backward()
                    opt.step()
                    epoch_loss.update(loss.item(), n=bs)
        
        LOG.info(f"Epoch {epoch+1}/{hparams.num_epochs} - "
                 f"avg_loss: {epoch_loss.avg:.4f}")
    
    LOG.info(f"Training complete. {len(adapter_map)} adapters trained.")
    return model, adapter_map


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk
