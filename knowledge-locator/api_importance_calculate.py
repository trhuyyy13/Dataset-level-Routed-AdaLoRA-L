from importance_calculate import APILayerImportance
import json
from tqdm import tqdm
import os
import torch

MODEL_PATH = ["deepseek-ai/deepseek-coder-1.3b-base", 'Qwen/Qwen2.5-Coder-3B', 'bigcode/starcoder2-3b']
MODEL_NAME = ['deepseek-1.3b', 'qwencoder-3b', 'starcoder-3b']

OUTPUT_PATH = 'knowledge-locator/layer_importance/{llm}/importance.json'
DATA_PATH = 'data/EditDeprecatedAPI/{llm}/all.json'

if __name__ == '__main__':
    for i, model in enumerate(MODEL_PATH):
        ipt_calculator = APILayerImportance(model)
        results = {}
        with open(DATA_PATH.format(llm=MODEL_NAME[i]), 'r') as f:
            bench_data = json.load(f)
            
        for line in tqdm(bench_data, desc=model):
            reference = line['reference']
            input_code = line['probing input'] + reference
            target_api = line['expected call']
            api_name = line['replacement api']
            case_id = line['case-id']
            ipt_calculator.initialize_importance()
            try:
                layer_ranking = ipt_calculator.compute_layer_importance(input_code, target_api)
            except Exception as e:
                print(f'ERROR {e} in case {case_id}')
                continue
            if api_name not in results.keys():
                results[api_name] = {}
            results[api_name].update({case_id: layer_ranking})
            
        for api, cases in results.items():
            api_average_importance = []
            case_count = 0
            for _case_id, _ranking in cases.items():
                case_count += 1
                if api_average_importance == []:
                    api_average_importance = _ranking
                else:
                    tmp_ranking_dict = {k : v for (k, v) in _ranking}
                    api_average_importance = [(k, v + tmp_ranking_dict[k]) for (k, v) in api_average_importance]
                    
            api_average_importance = {k : v / case_count for (k, v) in api_average_importance}
            api_average_importance = sorted(
                api_average_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            results[api].update({'average': api_average_importance})
        
        os.makedirs(f'knowledge-locator/layer_importance/{MODEL_NAME[i]}', exist_ok=True)
        with open(OUTPUT_PATH.format(llm=MODEL_NAME[i]), 'w') as f:
            json.dump(results, f, indent=4)
                
        del ipt_calculator
        torch.cuda.empty_cache()
        

