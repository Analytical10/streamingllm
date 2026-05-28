import torch

from .cache_utils import get_cache_seq_length


def aligned_position_ids_fn(kv_cache, window_length: int, device: str = "cuda"):
    cache_len = get_cache_seq_length(kv_cache)
    if cache_len < window_length:
        return torch.tensor([[cache_len]], dtype=torch.long, device=device)
    return torch.tensor([[window_length - 1]], dtype=torch.long, device=device)


def build_misaligned_position_ids_fn(device: str = "cuda"):
    step_state = {"step": 0}

    def _fn(_kv_cache):
        step_state["step"] += 1
        return torch.tensor([[step_state["step"]]], dtype=torch.long, device=device)

    return _fn
