import argparse
import csv
import math
import time
from collections import deque
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM

from streamingllm_experiment.cache_utils import (
    SinkCacheStrategy,
    WindowedCache,
    get_cache_seq_length,
)
from streamingllm_experiment.eval_utils import compute_streaming_loss
from streamingllm_experiment.pos_shift import enable_llama_pos_shift_attention
from streamingllm_experiment.tokenization import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp5 Table 2 attention sink ablation with a fixed KV cache size."
    )
    parser.add_argument("--model-name", default="/data/pretrained_models/Llama-2-7b-hf")
    parser.add_argument("--input", default="data/pg19_books_65133.txt")
    parser.add_argument("--total-cache-size", type=int, default=4096)
    parser.add_argument("--sink-sizes", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/exp5")
    parser.add_argument("--wandb-project", default="streamingllm-exp5")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--log-interval", type=int, default=1000)
    return parser.parse_args()


def build_strategy(total_cache_size: int, sink_tokens: int):
    if sink_tokens == 0:
        return "window", WindowedCache(max_len=total_cache_size)
    return "sink", SinkCacheStrategy(
        window_length=total_cache_size,
        num_sink_tokens=sink_tokens,
    )


def write_losses(path: Path, losses: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss"])
        for step, loss in enumerate(losses):
            writer.writerow([step, loss])


def append_summary(path: Path, row: dict) -> None:
    fieldnames = [
        "sink_tokens",
        "recent_tokens",
        "total_cache_size",
        "num_eval_tokens",
        "avg_nll",
        "overall_ppl",
        "elapsed_seconds",
    ]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def make_step_callback(smooth_window: int):
    recent = deque(maxlen=max(1, smooth_window))

    def step_callback(step: int, loss_value: float) -> None:
        recent.append(loss_value)
        loss_smooth = sum(recent) / len(recent)
        wandb.log(
            {
                "step": step,
                "loss": loss_value,
                "loss_smooth": loss_smooth,
                "log_ppl": loss_smooth,
            }
        )

    return step_callback


def main() -> None:
    args = parse_args()
    if args.total_cache_size <= 0:
        raise ValueError("--total-cache-size must be positive")
    for sink_tokens in args.sink_sizes:
        if sink_tokens < 0:
            raise ValueError("--sink-sizes cannot contain negative values")
        if sink_tokens >= args.total_cache_size:
            raise ValueError("Each sink size must be smaller than --total-cache-size")

    if not torch.cuda.is_available():
        raise RuntimeError("Exp5 is configured for CUDA, but torch.cuda.is_available() is False.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    device = "cuda"
    tokenizer = load_tokenizer(args.model_name)
    text = Path(args.input).read_text(encoding="utf-8")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    if args.max_tokens > 0:
        input_ids = input_ids[:, : args.max_tokens]
    input_ids = input_ids.to(device)
    print(f"Loaded input_ids with seq_len={input_ids.shape[1]}")

    print("Loading model...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"Model loaded in {time.time() - start:.2f}s")
    enable_llama_pos_shift_attention(model)

    for sink_tokens in args.sink_sizes:
        recent_tokens = args.total_cache_size - sink_tokens
        strategy_name, strategy = build_strategy(args.total_cache_size, sink_tokens)
        run_name = f"exp5_sink_{sink_tokens}_{recent_tokens}"
        print(f"Running {run_name} with total_cache_size={args.total_cache_size}")

        def aligned_position_ids_fn(cache):
            cache_len = get_cache_seq_length(cache)
            if cache_len < args.total_cache_size:
                return torch.tensor([[cache_len]], dtype=torch.long, device=device)
            return torch.tensor([[args.total_cache_size - 1]], dtype=torch.long, device=device)

        step_callback = None
        if args.wandb_mode != "disabled":
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model_name": args.model_name,
                    "input": args.input,
                    "total_cache_size": args.total_cache_size,
                    "sink_tokens": sink_tokens,
                    "recent_tokens": recent_tokens,
                    "max_tokens": args.max_tokens,
                    "smooth_window": args.smooth_window,
                    "strategy": strategy_name,
                },
                mode=args.wandb_mode,
                reinit=True,
            )
            step_callback = make_step_callback(args.smooth_window)

        group_start = time.time()
        losses = compute_streaming_loss(
            model,
            input_ids,
            kv_cache=strategy.init_cache(),
            custom_position_fn=aligned_position_ids_fn,
            cache_strategy=strategy,
            log_interval=args.log_interval,
            progress_desc=run_name,
            step_callback=step_callback,
        )
        elapsed_seconds = time.time() - group_start

        losses = [float(loss) for loss in losses]
        avg_nll = sum(losses) / len(losses) if losses else float("inf")
        overall_ppl = math.exp(avg_nll) if math.isfinite(avg_nll) else float("inf")
        num_eval_tokens = len(losses)

        losses_path = output_dir / f"losses_sink_{sink_tokens}.csv"
        write_losses(losses_path, losses)
        row = {
            "sink_tokens": sink_tokens,
            "recent_tokens": recent_tokens,
            "total_cache_size": args.total_cache_size,
            "num_eval_tokens": num_eval_tokens,
            "avg_nll": avg_nll,
            "overall_ppl": overall_ppl,
            "elapsed_seconds": elapsed_seconds,
        }
        append_summary(summary_path, row)

        print(
            f"{run_name}: avg_nll={avg_nll:.6f}, "
            f"overall_ppl={overall_ppl:.4f}, elapsed={elapsed_seconds:.2f}s"
        )
        if args.wandb_mode != "disabled":
            wandb.summary["avg_nll"] = avg_nll
            wandb.summary["overall_ppl"] = overall_ppl
            wandb.summary["num_eval_tokens"] = num_eval_tokens
            wandb.summary["elapsed_seconds"] = elapsed_seconds
            wandb.finish()

        del losses
        torch.cuda.empty_cache()

    print(f"Exp5 complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
