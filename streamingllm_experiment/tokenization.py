from typing import Optional

from transformers import AutoTokenizer


def load_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
