import argparse
import json
from pathlib import Path

from streamingllm_experiment.data.streameval import generate_streameval_sample
from streamingllm_experiment.paths import DATA_DIR
from streamingllm_experiment.tokenization import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--target-tokens", type=int, default=120000)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=str(DATA_DIR / "streameval_120k.jsonl"))
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_name)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            text, qa_pairs = generate_streameval_sample(
                tokenizer=tokenizer,
                target_tokens=args.target_tokens,
                seed=args.seed + i,
            )
            record = {"id": i, "text": text, "qa_pairs": qa_pairs}
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Saved {args.num_samples} samples to {output_path}")


if __name__ == "__main__":
    main()
