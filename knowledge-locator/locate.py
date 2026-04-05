import json
from transformers import AutoModelForCausalLM, AutoTokenizer

IPT_PATH = 'knowledge-locator/layer_importance/{llm}/importance.json'
OUT_PATH = 'knowledge-locator/layer_importance/{llm}/suggest_edit_layers.json'
ROUTED_OUT_PATH = 'knowledge-locator/layer_importance/{llm}/routed_layer_config.json'
MODEL_NAME = ['deepseek-1.3b', 'qwencoder-3b', 'starcoder-3b']  # 24, 36, 30
REMOVE_LAYER = [8, 8, 14]
LAYER_NUM = [8, 8, 10]


if __name__ == '__main__':
    for i, model in enumerate(MODEL_NAME):
        suggest_edit_layers = {}
        remove_layers = []
        with open(IPT_PATH.format(llm=model), 'r') as f:
            ipt_data = json.load(f)
        tmp_layer_count = {}
        for api_name, ipt in ipt_data.items():
            top_layers = []
            top_layers = ipt['average'][: len(ipt['average']) // 2]
            for (layer_idx, _) in top_layers:
                if layer_idx not in tmp_layer_count.keys():
                    tmp_layer_count.update({layer_idx: 1})
                else:
                    tmp_layer_count[layer_idx] += 1
        tmp_layer_count = sorted(
                tmp_layer_count.items(),
                key=lambda x: x[1],
                reverse=True
            )
        print(tmp_layer_count)
        for _ in tmp_layer_count:
            if len(remove_layers) >= REMOVE_LAYER[i]:
                remove_layers = set(remove_layers)
                break
            remove_layers.append(_[0])
        print(remove_layers)
        
        common_layers = sorted(list(remove_layers))
        
        for api_name, ipt in ipt_data.items():
            edit_layers = []
            for (layer_idx, _) in ipt['average']:
                if layer_idx not in remove_layers and len(edit_layers) < LAYER_NUM[i]:
                    edit_layers.append(layer_idx)
            suggest_edit_layers.update({api_name : edit_layers})
        
        # Original format (backward compatible)
        with open(OUT_PATH.format(llm=model), 'w') as f:
            json.dump(suggest_edit_layers, f, indent=4)
        
        # New routed format with common_layers
        routed_config = {
            "common_layers": common_layers,
            "api_specific_layers": suggest_edit_layers
        }
        with open(ROUTED_OUT_PATH.format(llm=model), 'w') as f:
            json.dump(routed_config, f, indent=4)
        
        print(f"[{model}] Common layers: {common_layers}")
        print(f"[{model}] APIs with specific layers: {len(suggest_edit_layers)}")