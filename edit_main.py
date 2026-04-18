from codellmeditor import (
    FTHyperParams, 
    PMETHyperParams,
    MALMENHyperParams,
    GraceHyperParams,
    NON_EDITHyperParams,
    ROMEHyperParams,
    MEMITHyperParams,
    AGraceHyperParams,
    LoRAHyperParams,
    AlphaEditHyperParams,
    REPLACE_APIHyperParams,
    RoutedAdaLoRAHyperParams
    )
from codellmeditor import BaseEditor, DAPIEDataset, EditTrainer, prepare_requests
import argparse
import logging
import gc
import torch
import os
import json


LOG = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', default='GRACE', type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--data_set', default='EditDeprecatedAPI', type=str)
    parser.add_argument('--model', default='qwencoder-3b', type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--generation_test_interval', default=0, type=int)
    parser.add_argument('--continue_from_run', default=None, type=str)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--log_level', default='INFO', type=str)
    parser.add_argument('--suggest_layers', default=None, type=str)
    parser.add_argument('--train_all_first', action='store_true', help="Tuỳ chọn Train hết data rồi test 1 lượt (dành cho baselines truyền thống)")

    args = parser.parse_args()
    log_level = logging.INFO
    if args.log_level == 'DEBUG':
        log_level = logging.DEBUG
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = log_level)
    train_datas = None
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'AGRACE':
        editing_hparams = AGraceHyperParams
    elif args.editing_method == 'PMET':
        editing_hparams = PMETHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LORA' or args.editing_method == 'ADALORA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'ROUTED_ADALORA':
        editing_hparams = RoutedAdaLoRAHyperParams
    elif args.editing_method == 'MALMEN':
        editing_hparams = MALMENHyperParams
        train_datas = DAPIEDataset(f"{args.data_dir}/{args.data_set}", data_type='train', model=args.model)
    elif args.editing_method == 'NON_EDIT':
        editing_hparams = NON_EDITHyperParams
    elif args.editing_method == 'ALPHAEDIT':
        editing_hparams = AlphaEditHyperParams
    elif args.editing_method == 'REPLACE_API':
        editing_hparams = REPLACE_APIHyperParams
    else:
        raise NotImplementedError
    
    hparams_dir = os.path.join('hparams', args.editing_method, args.model)
    hparams = editing_hparams.from_hparams(hparams_dir)


    if args.editing_method in ['MALMEN', 'AGRACE']:
        train_datas = DAPIEDataset(f"{args.data_dir}/{args.data_set}", data_type='train', model=args.model, tokenizer_name=hparams.model_name)
        test_datas = DAPIEDataset(f"{args.data_dir}/{args.data_set}", data_type='test', model=args.model)
    else:
        test_datas = DAPIEDataset(f"{args.data_dir}/{args.data_set}", args.model)
    
    test_datas = prepare_requests(test_datas, hparams.model_name, args.data_set, args.editing_method)
    if args.ds_size is not None:
        test_datas = test_datas.from_list(test_datas[:args.ds_size])
    if args.editing_method == 'MALMEN' and hparams.archive is None:
        train_datas = prepare_requests(train_datas, hparams.model_name, args.data_set, args.editing_method)
        trainer = EditTrainer(
            config=hparams,
            train_set=train_datas,
            val_set=test_datas
        )
        trainer.run()
        hparams.archive = trainer.save_path
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
    print(hparams)
    suggest_edit_layers = args.suggest_layers
    editor = BaseEditor.from_hparams(hparams, args.data_set)
    if suggest_edit_layers is not None:
        with open(args.suggest_layers, 'r') as f:
            suggest_edit_layers = json.load(f)
    if args.editing_method == 'ROUTED_ADALORA':
        # Routed AdaLoRA-L: dataset-level training with per-API adapters
        if suggest_edit_layers is None:
            raise ValueError("ROUTED_ADALORA requires --suggest_layers pointing to routed_layer_config.json")
        # Support both formats: routed config (with common_layers) or legacy (flat dict)
        if 'api_specific_layers' in suggest_edit_layers:
            layer_config = suggest_edit_layers
        else:
            layer_config = {"common_layers": [], "api_specific_layers": suggest_edit_layers}
        metrics, edited_model, _ = editor.edit_dataset_level(
            requests=test_datas,
            data_set_name=args.data_set,
            layer_config=layer_config,
            generation_test_interval=args.generation_test_interval,
            continue_from_run=args.continue_from_run,
        )
    elif args.train_all_first:
        metrics, edited_model, _ = editor.edit_dataset_level_baselines(
            requests = test_datas,
            data_set_name = args.data_set,
            generation_test_interval = args.generation_test_interval,
            continue_from_run = args.continue_from_run
        )
    elif args.editing_method in ['MALMEN', 'AGRACE']:
        train_datas = prepare_requests(train_datas, hparams.model_name, args.data_set, args.editing_method)
        metrics, edited_model, _ = editor.edit(
            requests = test_datas,
            keep_original_weight = True,
            generation_test_interval = args.generation_test_interval,
            data_set_name = args.data_set,
            continue_from_run = args.continue_from_run,
            another_part_data=train_datas # Used for retrieving portability data
        )
    else:
        metrics, edited_model, _ = editor.edit(
            requests = test_datas,
            keep_original_weight = True,
            generation_test_interval = args.generation_test_interval,
            data_set_name = args.data_set,
            suggest_edit_layers=suggest_edit_layers,
            continue_from_run = args.continue_from_run
        )
