import json
import typing
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
from typing import Dict, List
import os

def prepare_requests(datas, model_name, dataset_name, edit_method):
    if 'CodeLlama-7b-Python-hf' in model_name:
        model_name = 'CodeLlama-7b-Python-hf'
    elif 'deepseek-coder-1.3b-base' in model_name:
        model_name = 'deepseek-coder-1.3b-base'
    elif 'starcoder2-3b' in model_name:
        model_name = 'starcoder2-3b'
    elif 'Qwen2.5-Coder-3B' in model_name:
        model_name = 'Qwen2.5-Coder-3B'
    elif 'codegemma-2b' in model_name:
        model_name = 'codegemma-2b'
    else:
        raise NotImplementedError(f'answers of {model_name} not provided')
    if dataset_name == "EditDeprecatedAPI":
        requests = [{
            'case_id': data['case-id'],
            'prompt': data['probing input'],
            'target_new': data['reference'],
            'rephrase_prompt': data['rephrase'],
            'rephrase_target_new': data['rephrase_reference'],
            'reference_dict': data['reference dict'],
            'alias_dict': data['alias dict'],
            'rephrase_reference_dict': data['reference dict'] | data['rephrase_reference_dict'],
            'new_api': [[data['replacement api']]],
            'specificity': {'prompts': [item['probing input'] for item in data['Specificity-SimilarContext']],
                            'ground_truth': [item['prediction'] for item in data['Specificity-SimilarContext']],
                            'pred-api': [item['pred-api'] for item in data['Specificity-SimilarContext']]},
            "portability": data['portability'],
            "target_api": data['replacement api'],
            "probing_predictions": data['probing predictions'][0][0],
            "api_predicted": data['probing predictions'][0][1],
            "deprecated_api": data['deprecated api'],
            "expected_call": data['expected call'],
        } for data in datas]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implement!")

    return datas.from_list(requests)

def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict

class DAPIEDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        model: str,
        data_type: str = 'all',
        tokenizer_name: str = None,
    ):  
        self.data_dir = data_dir
        print(data_dir)
        if "DeprecatedAPI" not in data_dir:
            raise ValueError(f"Make sure data path is correct, you current data path is {data_dir}")
        else:
            self.data_name = "DeprecatedAPI"
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileExistsError(f"{data_dir} does not exist.")
        
        self.models = os.listdir(data_dir)
        
        if model not in self.models:
            raise ValueError(f"Do not support model {model}. The currently supported models are {self.models}")
        
        self.data = []
        with open(os.path.join(os.path.join(data_dir, model, f'{data_type}.json')), 'r') as f:
            data = json.load(f)
        self.data = data
            
        if tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def from_list(self, data):
        self.data = data
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

    def collate_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        rephrase_trg = [b["rephrase_target_new"] for b in batch]
        rephrase = [b["rephrase_prompt"] for b in batch]

        specs = [
                prompt 
                for b in batch
                for prompt in b["specificity"]["prompts"] 
                ]
        specs_ans = [
                prompt 
                for b in batch
                for prompt in b["specificity"]["ground_truth"] 
                ]
        src = [src_ + '\n' + trg_ for src_, trg_ in zip(src, trg)]
        rephrase = [rephrase_ + ' ' + r_trg_ for rephrase_, r_trg_ in zip(rephrase, rephrase_trg)]
        specs = [spec_ + ' ' + spec_ans_ for spec_, spec_ans_ in zip(specs, specs_ans)]
        
        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "rephrase": rephrase,
            }.items()
            for k2, v2 in self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
            ).items()
        }

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"])

        edit_inner["labels"] = edit_labels

        edit_rephrase = {}
        edit_rephrase["input_ids"] = batches["rephrase_input_ids"]
        edit_rephrase["attention_mask"] = batches["rephrase_attention_mask"]
        edit_rephrase["labels"] = edit_labels

        # loc
        loc = dict(
            self.tokenizer(
                specs[:2],
                return_tensors="pt",
                padding=True,
            )
        )

        loc_ans = dict(
            self.tokenizer(
                specs_ans[:2],
                return_tensors="pt",
                padding=True,
            )
        )
        loc["decoder_attention_mask"] = loc_ans["attention_mask"]
        loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])


        batch = {
            "edit_inner": edit_inner,
            "edit_rephrase": edit_rephrase,
            "loc": loc,
            "raw": batch,
        }

        return dict_to(batch, "cuda:0")
    