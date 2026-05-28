import argparse
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM

from streamingllm_experiment.cache_utils import SinkCacheStrategy
from streamingllm_experiment.eval_utils import greedy_generate_with_cache
from streamingllm_experiment.logging_utils import save_csv
from streamingllm_experiment.paths import OUTPUTS_DIR
from streamingllm_experiment.tokenization import load_tokenizer
from streamingllm_experiment.data.longbench import load_longbench_jsonl


def normalize_text(text: str) -> List[str]:
    return text.lower().split()


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred)
    gold_tokens = normalize_text(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def truncate_1750_1750(tokenizer, context: str, max_len: int = 3500):
    ids = tokenizer.encode(context)
    head = ids[: max_len // 2]
    tail = ids[-max_len // 2 :]
    return tokenizer.decode(head + tail, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--input", required=True, help="Path to LongBench NarrativeQA JSONL")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--window-length", type=int, default=3500)
    parser.add_argument("--num-sink-tokens", type=int, default=4)
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "exp3b"))
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto"
    )

    rows = []
    for i, sample in enumerate(load_longbench_jsonl(Path(args.input))):
        if i >= args.max_samples:
            break
        context = sample.get("context") or sample.get("document") or ""
        question = sample.get("question") or ""
        answers = sample.get("answers") or []
        gold = answers[0] if answers else ""

        # Baseline truncation
        truncated_context = truncate_1750_1750(tokenizer, context, max_len=3500)
        prompt = truncated_context + "\nQuestion: " + question + "\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        gen_ids, _ = greedy_generate_with_cache(
            model, input_ids, max_new_tokens=args.max_new_tokens, kv_cache=None, eos_token_id=tokenizer.eos_token_id
        )
        pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        f1_trunc = f1_score(pred, gold)

        # Streaming 4+3496
        sink_cache = SinkCacheStrategy(
            window_length=args.window_length, num_sink_tokens=args.num_sink_tokens
        ).init_cache()
        context_ids = tokenizer.encode(context, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(input_ids=context_ids, past_key_values=sink_cache, use_cache=True)
            sink_cache = _.past_key_values
        q_ids = tokenizer.encode("\nQuestion: " + question + "\nAnswer:", return_tensors="pt").to(device)
        gen_ids, sink_cache = greedy_generate_with_cache(
            model, q_ids, max_new_tokens=args.max_new_tokens, kv_cache=sink_cache, eos_token_id=tokenizer.eos_token_id
        )
        pred_stream = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        f1_stream = f1_score(pred_stream, gold)

        # Fixed sink 1750+1750 (use num_sink_tokens as 1750 in tokens)
        fixed_sink_cache = SinkCacheStrategy(window_length=3500, num_sink_tokens=1750).init_cache()
        with torch.no_grad():
            _ = model(input_ids=context_ids, past_key_values=fixed_sink_cache, use_cache=True)
            fixed_sink_cache = _.past_key_values
        gen_ids, fixed_sink_cache = greedy_generate_with_cache(
            model, q_ids, max_new_tokens=args.max_new_tokens, kv_cache=fixed_sink_cache, eos_token_id=tokenizer.eos_token_id
        )
        pred_fixed = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        f1_fixed = f1_score(pred_fixed, gold)

        rows.append(
            {
                "id": i,
                "f1_trunc": f1_trunc,
                "f1_stream": f1_stream,
                "f1_fixed_sink": f1_fixed,
            }
        )

    save_csv(Path(args.output_dir) / "longbench_results.csv", rows)


if __name__ == "__main__":
    main()
