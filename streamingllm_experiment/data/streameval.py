import random
from typing import Dict, List, Tuple


def build_kv_line(line_idx: int, value: int) -> str:
    return f"The REGISTER_CONTENT in line {line_idx} is <{value}>"


def build_query_line(line_idx: int) -> str:
    return f"Query: What is the REGISTER_CONTENT in line {line_idx}?"


def generate_streameval_sample(
    tokenizer,
    target_tokens: int,
    seed: int,
    kv_interval: int = 10,
    query_offset: int = 20,
) -> Tuple[str, List[Dict]]:
    rng = random.Random(seed)
    lines: List[str] = []
    qa_pairs: List[Dict] = []
    line_idx = 0
    while True:
        line_idx += 1
        if line_idx % kv_interval == 0:
            value = rng.randint(1, 99999)
            kv_line = build_kv_line(line_idx, value)
            lines.append(kv_line)
            query_line_idx = line_idx + query_offset
            qa_pairs.append({"line_idx": line_idx, "answer": f"<{value}>"})
            lines.append(build_query_line(line_idx))
        else:
            lines.append(f"Line {line_idx}: filler text.")

        token_count = len(tokenizer.encode("\n".join(lines)))
        if token_count >= target_tokens:
            break

    return "\n".join(lines), qa_pairs
