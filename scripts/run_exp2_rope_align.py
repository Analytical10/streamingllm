import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from streamingllm_experiment.cache_utils import WindowedCache
from streamingllm_experiment.eval_utils import compute_streaming_loss, losses_to_ppl, sliding_window_mean
from streamingllm_experiment.logging_utils import save_csv
from streamingllm_experiment.paths import OUTPUTS_DIR
from streamingllm_experiment.plotting import plot_lines
from streamingllm_experiment.position_utils import aligned_position_ids_fn, build_misaligned_position_ids_fn
from streamingllm_experiment.tokenization import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--input", default="data/pg19_20k.txt")
    parser.add_argument("--window-length", type=int, default=2048)
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "exp2"))
    parser.add_argument("--max-tokens", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_name)
    text = Path(args.input).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt")
    if args.max_tokens > 0:
        input_ids = input_ids[:, : args.max_tokens]
    input_ids = input_ids.to(device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto"
    )

    aligned_cache = WindowedCache(max_len=args.window_length)
    kv_cache = aligned_cache.init_cache()
    aligned_losses = compute_streaming_loss(
        model,
        input_ids,
        kv_cache=kv_cache,
        custom_position_fn=lambda cache: aligned_position_ids_fn(cache, args.window_length, device),
        cache_strategy=aligned_cache,
    )

    misaligned_cache = WindowedCache(max_len=args.window_length)
    kv_cache = misaligned_cache.init_cache()
    misaligned_fn = build_misaligned_position_ids_fn(device=device)
    misaligned_losses = compute_streaming_loss(
        model,
        input_ids,
        kv_cache=kv_cache,
        custom_position_fn=misaligned_fn,
        cache_strategy=misaligned_cache,
    )

    aligned_smoothed = sliding_window_mean(aligned_losses, args.smooth_window)
    misaligned_smoothed = sliding_window_mean(misaligned_losses, args.smooth_window)
    aligned_ppl = losses_to_ppl(aligned_smoothed)
    misaligned_ppl = losses_to_ppl(misaligned_smoothed)

    save_csv(
        Path(args.output_dir) / "aligned.csv",
        [
            {"step": i, "loss": aligned_losses[i], "loss_smooth": aligned_smoothed[i], "ppl": aligned_ppl[i]}
            for i in range(len(aligned_losses))
        ],
    )
    save_csv(
        Path(args.output_dir) / "misaligned.csv",
        [
            {
                "step": i,
                "loss": misaligned_losses[i],
                "loss_smooth": misaligned_smoothed[i],
                "ppl": misaligned_ppl[i],
            }
            for i in range(len(misaligned_losses))
        ],
    )

    x = list(range(len(aligned_losses)))
    plot_lines(
        x=x,
        ys=[aligned_ppl, misaligned_ppl],
        labels=["aligned", "misaligned"],
        title="Experiment 2 RoPE alignment",
        x_label="Token index",
        y_label="PPL",
        output_path=Path(args.output_dir) / "ppl_alignment.png",
    )


if __name__ == "__main__":
    main()
