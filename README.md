# EDAPIBench

## Environment Setup
Install the required dependencies via `pip` using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
## Data Preparation
The benchmark is already located in the `data/EditDeprecatedAPI` directory. No additional data downloading or preprocessing is required.

## Baseline Evaluation
Run the following command to perform baseline evaluation:
```bash
python edit_main.py \
     --editing_method=[method_name] \
     --model=[model_name] \
```
### Notes on Specific Methods
1. **AGRACE**: Before executing the above `edit_main.py` script, you need to first train its encoder by running:
   ```bash
   python train_agrace_encoder.py \
          --model [model_name] \
          --data_set EditDeprecatedAPI \
          --data_dir [dataset_path] \
          --save_dir [result_save_path] \
          --mode mean
   ```
2. **MALMEN & AGRACE**: For these two methods, you need to split the benchmark into training set and test set first.

## Evaluation for AdaLoRA-L
1. First, execute the following bash script:
   ```bash
   sh locate_API_layers.sh
   ```
2. Then run the evaluation command:
   ```bash
   python edit_main.py \
     --editing_method=ADALORA \
     --model=[model_name] \
     --suggest_layers=knowledge-locator/layer_importance/[model_name]/suggest_edit_layers.json
   ```# Dataset-level-Routed-AdaLoRA-L
