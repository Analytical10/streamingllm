from typing import Iterable, List, Tuple

from datasets import load_dataset


def iter_pg19_texts(split: str = "test", streaming: bool = False) -> Iterable[str]:
    dataset = load_dataset("pg19", split=split, streaming=streaming)
    for item in dataset:
        yield item["text"]


def concat_until_target_incremental(
    texts: Iterable[str],
    target_tokens: int,
    tokenizer,
    max_books: int = 0,
) -> Tuple[str, int, int]:
    combined: List[str] = []
    total_tokens = 0
    num_used = 0
    for text in texts:
        if max_books > 0 and num_used >= max_books:
            break
        num_used += 1
        combined.append(text)
        total_tokens += len(tokenizer.encode(text))
        print(f"After adding book {num_used}, total tokens: {total_tokens}")
        if total_tokens >= target_tokens:
            break
    return "\n".join(combined), total_tokens, num_used
