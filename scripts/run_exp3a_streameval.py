import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from streamingllm_experiment.cache_utils import SinkCacheStrategy
from streamingllm_experiment.eval_utils import greedy_generate_with_cache
from streamingllm_experiment.logging_utils import save_csv
from streamingllm_experiment.paths import OUTPUTS_DIR
from streamingllm_experiment.plotting import plot_lines
from streamingllm_experiment.tokenization import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--input", default="data/streameval_120k.jsonl")
    parser.add_argument("--window-length", type=int, default=2048)
    parser.add_argument("--num-sink-tokens", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "exp3a"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto"
    )

    results = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record["text"]
            qa_pairs = record["qa_pairs"]

            kv_cache = SinkCacheStrategy(
                window_length=args.window_length, num_sink_tokens=args.num_sink_tokens
            ).init_cache()

            total_tokens = 0
            correct = 0
            queries_seen = 0

            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue
                line_ids = tokenizer.encode(line + "\n", return_tensors="pt").to(device)

                if line.startswith("Query:"):
                    queries_seen += 1
                    gen_ids, kv_cache = greedy_generate_with_cache(
                        model,
                        input_ids=line_ids,
                        max_new_tokens=args.max_new_tokens,
                        kv_cache=kv_cache,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    answer = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
                    expected = qa_pairs[queries_seen - 1]["answer"] if qa_pairs else ""
                    is_correct = expected in answer
                    if is_correct:
                        correct += 1
                    results.append(
                        {
                            "sample_id": record["id"],
                            "query_index": queries_seen,
                            "token_index": total_tokens,
                            "pred": answer,
                            "expected": expected,
                            "correct": int(is_correct),
                        }
                    )
                else:
                    with torch.no_grad():
                        outputs = model(input_ids=line_ids, past_key_values=kv_cache, use_cache=True)
                        kv_cache = outputs.past_key_values
                        total_tokens += line_ids.shape[1]

    output_path = Path(args.output_dir) / "streameval_results.csv"
    save_csv(output_path, results)

    if results:
        token_index = [r["token_index"] for r in results]
        acc_prefix = []
        correct = 0
        for i, r in enumerate(results, start=1):
            correct += r["correct"]
            acc_prefix.append(correct / i)

        plot_lines(
            x=token_index,
            ys=[acc_prefix],
            labels=["exact_match"],
            title="Experiment 3a StreamEval accuracy",
            x_label="Token index",
            y_label="Exact match",
            output_path=Path(args.output_dir) / "accuracy_curve.png",
        )


if __name__ == "__main__":
    main()
