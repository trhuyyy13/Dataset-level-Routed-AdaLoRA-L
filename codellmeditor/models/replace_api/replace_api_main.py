from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from .replace_api_hparams import REPLACE_APIHyperParams
from codellmeditor.evaluate.evaluate_utils import (
    batch_generate,
    extract_first_statement,
)
from codellmeditor.util.source_utils import clean_pred, extract_apis_in_first_stmt
import re

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
KZ_CACHE= {}

def index_of_api(pred:str, target_apis, ref_dict, alias_dict):
    pkg_as = dict()
    for alias, name in alias_dict.items():
        alias_parts = alias.split(".")
        name_parts = name.split(".")
        while len(alias_parts) > 0 and len(name_parts) > 0 and alias_parts[-1] == name_parts[-1]:
            alias_parts.pop()
            name_parts.pop()
        pkg_alias, pkg_name = ".".join(alias_parts), ".".join(name_parts)
        if pkg_alias != pkg_name:
            pkg_as[pkg_alias] = pkg_name


    for mobj in re.finditer(r"([\w\.]+)\s*\(", pred):
        api = mobj.group(1).strip()
        if api == "":
            continue
        parts = api.split('.')
        if len(parts) == 2 and parts[0] in ref_dict:
            api = f"{ref_dict[parts[0]]}.{parts[1]}"
        if api in alias_dict:
            api = alias_dict[api]
        else:
            for pkg_alias, pkg_name in pkg_as.items():
                if api.startswith(f"{pkg_alias}."):
                    api = api.replace(f"{pkg_alias}.", f"{pkg_name}.")
                    break
        if api in target_apis:
            idx = pred.index(mobj.group(0))
            return idx
    return 0

def prcess_data(item):
    _pred, _api_preds = item["probing_predictions"], item["api_predicted"]
    target_apis = set(item['deprecated_api']) & set(_api_preds)
    idx = index_of_api(_pred, target_apis, item["reference_dict"], item["alias_dict"])
    item["replace_prompt"] = item["prompt"] + _pred[:idx] + item["expected_call"]

    return item

def prcess_rephrased_data(item, rephrase_expect_call, rephrase_pred, rephrase_api_preds):
    target_apis = set(item['deprecated_api']) & set(rephrase_api_preds)
    idx = index_of_api(rephrase_pred, target_apis, item["rephrase_reference_dict"], item["alias_dict"])
    item["replace_rephrase_prompt"] = item["rephrase_prompt"] + rephrase_pred[:idx] + rephrase_expect_call

    return item

def apply_replace_api_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: REPLACE_APIHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    prcess_data(requests[0])
    if requests[0]["portability"] != "":
        prcess_data(requests[0]["portability"])
    gen_strs = batch_generate(model, tok, requests[0]['rephrase_prompt'], max_length=50)
    _preds = [clean_pred(p) for p in gen_strs]
    gen_strs = [extract_first_statement(p, False) for p in _preds]
    gen_apis_rephrase = extract_apis_in_first_stmt(_preds[0], requests[0]['rephrase_reference_dict'], requests[0]['alias_dict'])
    if len(set(requests[0]['deprecated_api']) & set(gen_apis_rephrase)) > 0:
        rephrase_expect_call_idx = index_of_api(requests[0]['rephrase_target_new'], requests[0]['new_api'][0], requests[0]["rephrase_reference_dict"], requests[0]["alias_dict"])
        left_paren_idx = requests[0]['rephrase_target_new'].find('(', rephrase_expect_call_idx)
        rephrase_expect_call = requests[0]['rephrase_target_new'][rephrase_expect_call_idx:left_paren_idx]
        prcess_rephrased_data(requests[0], rephrase_expect_call, gen_strs[0], gen_apis_rephrase)
    else:
        # If no deprecated API detected in rephrase, use the original rephrase_prompt
        requests[0]["replace_rephrase_prompt"] = requests[0]["rephrase_prompt"]

    return model, {}
    

