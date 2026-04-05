from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch


def _setup_tokenizer(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def init_codellama7b(
    model_path="codellama/CodeLlama-7b-Python-hf",
    device="cuda",
    torch_dtype=torch.float16,
):
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        use_fast=False,
    )
    tokenizer = _setup_tokenizer(tokenizer)
    return model, tokenizer


def init_deepseek1b(
    model_path="deepseek-ai/deepseek-coder-1.3b-base",
    device="cuda",
    torch_dtype=torch.float16,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False,
    )
    tokenizer = _setup_tokenizer(tokenizer)
    return model, tokenizer


def init_codegemma2b(
    model_path="google/codegemma-2b",
    device="cuda",
    torch_dtype=torch.float16,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False,
    )
    tokenizer = _setup_tokenizer(tokenizer)
    return model, tokenizer


def init_starcoder3b(
    model_path="bigcode/starcoder2-3b",
    device="cuda",
    torch_dtype=torch.float16,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False,
    )
    tokenizer = _setup_tokenizer(tokenizer)
    return model, tokenizer


def init_qwencoder3b(
    model_path="Qwen/Qwen2.5-Coder-3B",
    device="cuda",
    torch_dtype=torch.float16,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False,
    )
    tokenizer = _setup_tokenizer(tokenizer)
    return model, tokenizer


MODEL_FACTORY = {
    "codellama/CodeLlama-7b-Python-hf": init_codellama7b,
    "deepseek-ai/deepseek-coder-1.3b-base": init_deepseek1b,
    "bigcode/starcoder2-3b": init_starcoder3b,
    "Qwen/Qwen2.5-Coder-3B": init_qwencoder3b,
    "google/codegemma-2b": init_codegemma2b,
}