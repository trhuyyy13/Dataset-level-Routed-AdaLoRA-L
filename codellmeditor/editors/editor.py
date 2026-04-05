import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from datetime import datetime
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
# from accelerate import Accelerator
from ..models import MODEL_FACTORY
from ..util.globals import *
from .singleton_editor import SingletonEditor
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, MATCH_METRICS
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

LOG = logging.getLogger(__name__)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def make_logs(log_name = None):
    if log_name is None:
        log_name = 'run.log'
    f_h, s_h = get_handler('logs', log_name)

    LOG.handlers.clear()
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False
    
    for handler in [f_h, s_h]:
        handler.setLevel(logging.DEBUG)
        LOG.addHandler(handler)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)
  
class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams,  data_set_name):

        return cls(hparams, data_set_name)

    def __init__(self,
                hparams: HyperParams,
                data_set_name,
                 ):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        time_now = datetime.now().strftime("%m-%d-%H-%M")
        make_logs(f"{self.alg_name}_{self.model_name.split('/')[-1]}_{data_set_name}_{time_now}.log")

        LOG.debug("Instantiating model")
        self.device_map = 'auto' if hparams.model_parallel else None
        self.torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
        self.model, self.tok = MODEL_FACTORY[self.model_name](torch_dtype=self.torch_dtype)
        self.cuda_num = hparams.device
        self.model.to(f'cuda:{self.cuda_num}')
        self.hparams = hparams

    def edit(self,
             requests,
             data_set_name,
             generation_test_interval: Optional[int] = 0,
             keep_original_weight=False,
             continue_from_run = None,
             suggest_edit_layers=None,
             another_part_data = None,
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        edited_model, weights_copy = None, None
        results_dir = Path(f"./results/{data_set_name}/{self.alg_name}/{self.model_name.split('/')[-1]}")
        if continue_from_run:
            run_id = continue_from_run
            run_file = results_dir / f"{continue_from_run}.json"
            if run_file.exists():
                all_metrics = json.load(open(run_file))
                computed_cases = [item['case_id'] for item in all_metrics]
                output_file = run_file
        else:
            if results_dir.exists():
                id_list = [
                    int(str(x).split("_")[-1][:3])
                    for x in results_dir.iterdir()
                    if str(x).split("_")[-1][:3].isnumeric()
                ]
                run_id = 0 if not id_list else max(id_list) + 1
            else:
                run_id = 0
                os.makedirs(results_dir)  
            all_metrics = []    
            output_file = results_dir / f"run_{str(run_id).zfill(3)}.json"
        LOG.info(f"Results will be stored in {output_file}")
        if data_set_name == "EditConala":
            codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", local_files_only=True)
        else:
            codebert_tokenizer = None
        for index, request in tqdm(enumerate(requests), total=len(requests)):
            if request["case_id"] in ['']:
                continue
            request = request.copy()
            if 'MALMEN' in self.alg_name or 'AGRACE' in self.alg_name:
                if request["portability"] != "":
                    for line in requests:
                        if line['case_id'] == request["portability"]:
                            request["portability"] = line
                            break
                    for line in another_part_data:
                        if line['case_id'] == request["portability"]:
                            request["portability"] = line
                            break
            else:
                if request["portability"] != "":
                    for line in requests:
                        if line['case_id'] == request["portability"]:
                            request["portability"] = line
                            break
            if continue_from_run:
                if request["case_id"] in computed_cases:
                    LOG.debug(f"Case {request['case_id']} already exists.")
                    continue
            if suggest_edit_layers is not None and request["target_api"] in suggest_edit_layers.keys():
                self.hparams.layers = suggest_edit_layers[request["target_api"]]
            if 'LoRA' in self.alg_name and suggest_edit_layers is not None:
                self.model = self.model.to('cpu')
                del self.model
                torch.cuda.empty_cache()
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.torch_dtype, device_map=self.device_map, local_files_only=True)
                self.model.to(f'cuda:{self.cuda_num}')
                LOG.info(f'Edit layers: {self.hparams.layers} in case {request["case_id"]}')
            start = time()
            try:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                    LOG.error(f"CUDA OOM at case {request['case_id']}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if 'GRACE' in self.alg_name and keep_original_weight:
                        with torch.no_grad():
                            try:
                                weights_copy() # unpatch_fn
                            except Exception as e:
                                LOG.error(f"weights_copy ERROR at case {request['case_id']}: {e}")
                    else:
                        with torch.no_grad():
                            try:
                                for k, v in weights_copy.items():
                                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                            except Exception as e:
                                LOG.error(f"weights_copy ERROR at case {request['case_id']}: {e}")
                    continue
                else:
                    with open(output_file, 'w') as f:  
                        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
                    raise RuntimeError(e)
                
            exec_time = time() - start
            LOG.info(f"Execution {request['case_id']} editing took {exec_time}")
            try:
                
                all_metrics.append({
                    'case_id': request['case_id'],
                    # "requested_rewrite": request,
                    "time": exec_time,
                    "max_memory": torch.cuda.max_memory_allocated(f"cuda:{self.hparams.device}")/ 1024**2,
                    "post": compute_edit_quality(edited_model, self.tok, request, test_generation = (generation_test_interval % (1 + index) == 0), tokenizer_for_fluency=codebert_tokenizer),
                })
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    LOG.error(f"CUDA OOM at case {request['case_id']}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    if 'GRACE' in self.alg_name and keep_original_weight:
                        with torch.no_grad():
                            try:
                                weights_copy() # unpatch_fn
                            except Exception as e:
                                LOG.error(f"weights_copy ERROR at case {request['case_id']}: {e}")
                    else:
                        with torch.no_grad():
                            try:
                                for k, v in weights_copy.items():
                                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                            except Exception as e:
                                LOG.error(f"weights_copy ERROR at case {request['case_id']}: {e}")
                                
                    continue
                else:
                    with open(output_file, 'w') as f:  
                        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
                    raise RuntimeError(e)
                
            torch.cuda.reset_peak_memory_stats(f"cuda:{self.hparams.device}")
            if 'GRACE' in self.alg_name and keep_original_weight:
                with torch.no_grad():
                    try:
                        weights_copy() # unpatch_fn
                    except Exception as e:
                        LOG.error(f"weights_copy ERROR at case {request['case_id']}: {e}")
            else:
                with torch.no_grad():
                    try:
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                    except Exception as e:
                        LOG.error(f"weights_copy ERROR at case {request['case_id']}: {e}")
            LOG.debug(
                    f"{request['case_id']} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[-1]} \n max memory: {torch.cuda.max_memory_allocated(f'cuda:{self.hparams.device}')/ 1024**2:.2f}M"
                )
            if (index+1)%10 == 9:
                with open(output_file, 'w') as f:  
                    json.dump(all_metrics, f, ensure_ascii=False, indent=4)
        
        with open(output_file, 'w') as f:  
            json.dump(all_metrics, f, ensure_ascii=False, indent=4)
        
        mean_metrics = dict()
        mean_metrics['run_id'] = run_id
        for metric in ['efficacy', 'generalization', 'portability', 'specificity']:
            mean_metrics[metric] = dict()
            for match_metric in MATCH_METRICS:
                mean_metrics[metric][match_metric] = (np.round(np.mean([item['post'][metric][match_metric] for item in all_metrics])*100, 2), np.round(np.std([item['post'][metric][match_metric] for item in all_metrics])*100, 2))
        ngram_entropys = []
        for item in all_metrics:
            if 'ngram_entropy' in item['post']:
                ngram_entropys.append(item['post']['ngram_entropy'])
        mean_metrics['fluency'] = (np.round(np.mean(ngram_entropys)*100, 2), np.round(np.std(ngram_entropys)*100, 2))
        mean_metrics["time"] = (np.round(np.mean([metric["time"] for metric in all_metrics]),3), np.round(np.std([metric["time"] for metric in all_metrics]),3))
        mean_metrics["max_memory"] = (np.round(np.mean([metric["max_memory"] for metric in all_metrics]),3), np.round(np.std([metric["max_memory"] for metric in all_metrics]),3))
        mean_metrics["hparams"] = str(self.hparams)
        mean_metrics_save_dir = results_dir / f"mean_run_{str(run_id).zfill(3)}.json"
        with open(mean_metrics_save_dir, 'w') as f:  
            json.dump(mean_metrics, f, ensure_ascii=False)
        LOG.info(f"Run {run_id}\nMetrics Summary: {mean_metrics}")
        LOG.info(self.hparams)

        return all_metrics, edited_model, weights_copy

    def edit_dataset_level(self,
             requests,
             data_set_name,
             layer_config: Dict,
             generation_test_interval: Optional[int] = 0,
             continue_from_run=None,
             ):
        """
        Dataset-level editing for Routed AdaLoRA-L:
        1. Train ALL APIs at once with per-API adapter routing
        2. Evaluate each case using its API-specific adapter
        """
        results_dir = Path(f"./results/{data_set_name}/{self.alg_name}/{self.model_name.split('/')[-1]}")
        if continue_from_run:
            run_id = continue_from_run
            run_file = results_dir / f"{continue_from_run}.json"
            if run_file.exists():
                all_metrics = json.load(open(run_file))
                computed_cases = [item['case_id'] for item in all_metrics]
                output_file = run_file
        else:
            if results_dir.exists():
                id_list = [
                    int(str(x).split("_")[-1][:3])
                    for x in results_dir.iterdir()
                    if str(x).split("_")[-1][:3].isnumeric()
                ]
                run_id = 0 if not id_list else max(id_list) + 1
            else:
                run_id = 0
                os.makedirs(results_dir)
            all_metrics = []
            output_file = results_dir / f"run_{str(run_id).zfill(3)}.json"
        LOG.info(f"Results will be stored in {output_file}")
        
        if data_set_name == "EditConala":
            codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", local_files_only=True)
        else:
            codebert_tokenizer = None

        # ── Phase 1: Train all APIs at once ──
        LOG.info("=" * 60)
        LOG.info("Phase 1: Dataset-level training with per-API routing")
        LOG.info("=" * 60)
        start_train = time()
        
        edited_model, adapter_map = self.apply_algo(
            self.model,
            self.tok,
            list(requests),
            self.hparams,
            layer_config=layer_config,
            copy=False,
            return_orig_weights=False,
        )
        
        train_time = time() - start_train
        LOG.info(f"Training completed in {train_time:.2f}s")
        LOG.info(f"Adapters created: {len(adapter_map)}")
        
        # ── Phase 2: Evaluate each case ──
        LOG.info("=" * 60)
        LOG.info("Phase 2: Evaluating each case")
        LOG.info("=" * 60)
        
        from ..models.routed_adalora.routed_adalora_main import sanitize_adapter_name
        
        for index, request in tqdm(enumerate(requests), total=len(requests)):
            if request["case_id"] in ['']:
                continue
            request = request.copy()
            
            # Resolve portability link
            if request["portability"] != "":
                for line in requests:
                    if line['case_id'] == request["portability"]:
                        request["portability"] = line
                        break
            
            if continue_from_run and request["case_id"] in computed_cases:
                LOG.debug(f"Case {request['case_id']} already exists.")
                continue
            
            # ── ROUTING: set the right adapter for this API ──
            api_name = request.get("target_api", "unknown")
            adapter_name = sanitize_adapter_name(api_name)
            
            if api_name in adapter_map:
                edited_model.set_adapter(adapter_map[api_name])
                LOG.debug(f"Case {request['case_id']}: using adapter '{adapter_map[api_name]}'")
            else:
                LOG.warning(f"Case {request['case_id']}: no adapter for API '{api_name}', using last active adapter")
            
            start = time()
            torch.cuda.reset_peak_memory_stats(f"cuda:{self.hparams.device}")
            
            try:
                gen_test_interval = generation_test_interval if index % generation_test_interval == 0 else -1
            except ZeroDivisionError:
                gen_test_interval = 0
            
            try:
                all_metrics.append({
                    'case_id': request['case_id'],
                    'time': time() - start,
                    'train_time': train_time,
                    'max_memory': torch.cuda.max_memory_allocated(f'cuda:{self.hparams.device}') / 1024**2,
                    'adapter': adapter_map.get(api_name, 'none'),
                    'post': compute_edit_quality(
                        edited_model, self.tok, request,
                        test_generation=gen_test_interval > 0,
                        tokenizer_for_fluency=codebert_tokenizer,
                    )
                })
            except Exception as e:
                LOG.error(f"Case {request['case_id']} error: {e}")
                with open(output_file, 'w') as f:
                    json.dump(all_metrics, f, ensure_ascii=False, indent=4)
                raise RuntimeError(e)
            
            LOG.debug(
                f"{request['case_id']} eval: {request['prompt'][:60]}... -> {request['target_new'][:40]}...\n"
                f"  {all_metrics[-1]}"
            )
            if (index + 1) % 10 == 0:
                with open(output_file, 'w') as f:
                    json.dump(all_metrics, f, ensure_ascii=False, indent=4)

        # ── Save results ──
        with open(output_file, 'w') as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=4)

        mean_metrics = dict()
        mean_metrics['run_id'] = run_id
        mean_metrics['train_time'] = train_time
        mean_metrics['num_adapters'] = len(adapter_map)
        for metric in ['efficacy', 'generalization', 'portability', 'specificity']:
            mean_metrics[metric] = dict()
            for match_metric in MATCH_METRICS:
                vals = [item['post'][metric][match_metric] for item in all_metrics]
                mean_metrics[metric][match_metric] = (
                    np.round(np.mean(vals) * 100, 2),
                    np.round(np.std(vals) * 100, 2)
                )
        ngram_entropys = [item['post']['ngram_entropy'] for item in all_metrics if 'ngram_entropy' in item['post']]
        mean_metrics['fluency'] = (
            np.round(np.mean(ngram_entropys) * 100, 2) if ngram_entropys else 0,
            np.round(np.std(ngram_entropys) * 100, 2) if ngram_entropys else 0
        )
        mean_metrics["time"] = (
            np.round(np.mean([m["time"] for m in all_metrics]), 3),
            np.round(np.std([m["time"] for m in all_metrics]), 3)
        )
        mean_metrics["max_memory"] = (
            np.round(np.mean([m["max_memory"] for m in all_metrics]), 3),
            np.round(np.std([m["max_memory"] for m in all_metrics]), 3)
        )
        mean_metrics["hparams"] = str(self.hparams)
        mean_metrics_save_dir = results_dir / f"mean_run_{str(run_id).zfill(3)}.json"
        with open(mean_metrics_save_dir, 'w') as f:
            json.dump(mean_metrics, f, ensure_ascii=False)
        LOG.info(f"Run {run_id}\nMetrics Summary: {mean_metrics}")
        LOG.info(self.hparams)

        return all_metrics, edited_model, adapter_map
