import argparse
from pathlib import Path

from streamingllm_experiment.data.pg19 import concat_until_target_incremental, iter_pg19_texts
from streamingllm_experiment.logging_utils import save_json
from streamingllm_experiment.paths import DATA_DIR
from streamingllm_experiment.tokenization import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="/data/pretrained_models/Llama-2-7b-hf")
    parser.add_argument("--target-tokens", type=int, default=20000)
    parser.add_argument("--num-books", type=int, default=3)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=str(DATA_DIR / "pg19_20k.txt"))
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_name)
    texts = iter_pg19_texts(split=args.split, streaming=args.streaming)
    combined, token_count, num_used = concat_until_target_incremental(
        texts, args.target_tokens, tokenizer, max_books=args.num_books
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined, encoding="utf-8")

    meta = {
        "model_name": args.model_name,
        "target_tokens": args.target_tokens,
        "actual_tokens": token_count,
        "num_books_used": num_used,
        "output": str(output_path),
    }
    save_json(output_path.with_suffix(".meta.json"), meta)
    print(f"Saved {output_path} with {token_count} tokens")


if __name__ == "__main__":
    main()
