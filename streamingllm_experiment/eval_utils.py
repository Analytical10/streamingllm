import math
from typing import Callable, Iterable, List, Optional

from tqdm import tqdm

import torch
import torch.nn as nn


def compute_streaming_loss(
    model,
    input_ids: torch.Tensor,
    kv_cache=None,
    custom_position_fn: Optional[Callable] = None,
    cache_strategy: Optional[Callable] = None,
    log_interval: int = 0,
    progress_desc: str = "streaming",
    step_callback: Optional[Callable[[int, float], None]] = None,
) -> List[float]:
    model.eval()
    losses: List[float] = []
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    seq_len = input_ids.shape[1]

    iterator = tqdm(range(seq_len - 1), desc=progress_desc)
    with torch.no_grad():
        for i in iterator:
            cur_token = input_ids[:, i : i + 1]
            next_token = input_ids[:, i + 1 : i + 2]

            position_ids = None
            if custom_position_fn is not None and kv_cache is not None:
                position_ids = custom_position_fn(kv_cache)

            outputs = model(
                input_ids=cur_token,
                past_key_values=kv_cache,
                position_ids=position_ids,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]
            kv_cache = outputs.past_key_values
            if cache_strategy is not None:
                kv_cache = cache_strategy.after_forward(kv_cache, i)

            step_loss = loss_fct(logits, next_token.view(-1))
            step_loss_value = float(step_loss.item())
            losses.append(step_loss_value)
            if step_callback is not None:
                step_callback(i, step_loss_value)
            if log_interval > 0 and (i + 1) % log_interval == 0:
                print(f"step {i + 1}: loss={step_loss_value:.6f}")

    return losses


def sliding_window_mean(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    out = []
    cumsum = 0.0
    for i, val in enumerate(values):
        cumsum += val
        if i >= window:
            cumsum -= values[i - window]
            out.append(cumsum / window)
        elif i == window - 1:
            out.append(cumsum / window)
    prefix = [out[0]] * (window - 1)
    return prefix + out


def losses_to_ppl(losses: Iterable[float]) -> List[float]:
    return [math.exp(l) for l in losses]


def greedy_generate_with_cache(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    kv_cache=None,
    eos_token_id: Optional[int] = None,
):
    model.eval()
    generated = []
    with torch.no_grad():
        cur_input = input_ids
        for _ in range(max_new_tokens):
            outputs = model(input_ids=cur_input, past_key_values=kv_cache, use_cache=True)
            kv_cache = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            cur_input = next_token
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
    if generated:
        return torch.cat(generated, dim=1), kv_cache
    return torch.empty((input_ids.size(0), 0), dtype=input_ids.dtype, device=input_ids.device), kv_cache
