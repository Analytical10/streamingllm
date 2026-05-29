import torch
from dataclasses import dataclass
from typing import Any, Optional

def get_cache_seq_length(kv_cache: Any) -> int:
    if kv_cache is None:
        return 0
    if hasattr(kv_cache, "get_seq_length"):
        return int(kv_cache.get_seq_length(layer_idx=0))
    if isinstance(kv_cache, (list, tuple)) and kv_cache:
        key = kv_cache[0][0]
        return int(key.shape[-2])
    return 0


def trim_past_key_values(kv_cache: Any, max_len: int) -> Any:
    if kv_cache is None:
        return None
    if hasattr(kv_cache, "key_cache") and hasattr(kv_cache, "value_cache"):
        for i in range(len(kv_cache.key_cache)):
            kv_cache.key_cache[i] = kv_cache.key_cache[i][..., -max_len:, :]
            kv_cache.value_cache[i] = kv_cache.value_cache[i][..., -max_len:, :]
        if hasattr(kv_cache, "_seen_tokens"):
            kv_cache._seen_tokens = max_len
        return kv_cache
    if isinstance(kv_cache, (list, tuple)):
        trimmed = []
        for key, value in kv_cache:
            trimmed.append((key[..., -max_len:, :], value[..., -max_len:, :]))
        return tuple(trimmed)
    return kv_cache


def trim_past_key_values_with_sink(kv_cache: Any, max_len: int, num_sink: int) -> Any:
    if kv_cache is None:
        return None
    if hasattr(kv_cache, "key_cache") and hasattr(kv_cache, "value_cache"):
        for i in range(len(kv_cache.key_cache)):
            if kv_cache.key_cache[i].shape[-2] <= max_len:
                continue
            recent_len = max_len - num_sink
            kv_cache.key_cache[i] = torch.cat([
                kv_cache.key_cache[i][..., :num_sink, :],
                kv_cache.key_cache[i][..., -recent_len:, :]
            ], dim=-2)
            kv_cache.value_cache[i] = torch.cat([
                kv_cache.value_cache[i][..., :num_sink, :],
                kv_cache.value_cache[i][..., -recent_len:, :]
            ], dim=-2)
        if hasattr(kv_cache, "_seen_tokens"):
            kv_cache._seen_tokens = max_len
        return kv_cache
    if isinstance(kv_cache, (list, tuple)):
        trimmed = []
        for key, value in kv_cache:
            if key.shape[-2] <= max_len:
                trimmed.append((key, value))
            else:
                recent_len = max_len - num_sink
                new_key = torch.cat([key[..., :num_sink, :], key[..., -recent_len:, :]], dim=-2)
                new_value = torch.cat([value[..., :num_sink, :], value[..., -recent_len:, :]], dim=-2)
                trimmed.append((new_key, new_value))
        return tuple(trimmed)
    return kv_cache


@dataclass
class CacheStrategy:
    name: str

    def init_cache(self):
        return None

    def after_forward(self, kv_cache, step: int):
        return kv_cache


@dataclass
class DenseCache(CacheStrategy):
    name: str = "dense"


@dataclass
class WindowedCache(CacheStrategy):
    max_len: int = 2048
    name: str = "window"

    def after_forward(self, kv_cache, step: int):
        return trim_past_key_values(kv_cache, self.max_len)


@dataclass
class SinkCacheStrategy(CacheStrategy):
    window_length: int = 2048
    num_sink_tokens: int = 4
    name: str = "sink"

    def after_forward(self, kv_cache, step: int):
        return trim_past_key_values_with_sink(kv_cache, self.window_length, self.num_sink_tokens)
