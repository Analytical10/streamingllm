import argparse
import math
import time
from collections import deque
from pathlib import Path

import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from streamingllm_experiment.cache_utils import DenseCache, SinkCacheStrategy, WindowedCache
from streamingllm_experiment.eval_utils import compute_streaming_loss, losses_to_ppl, sliding_window_mean
from streamingllm_experiment.tokenization import load_tokenizer

def run_recompute(
    model,
    input_ids,
    window_length: int,
    log_interval: int,
    step_callback,
):
    losses = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    seq_len = input_ids.shape[1]
    iterator = tqdm(range(seq_len - 1), desc="recompute")
    with torch.no_grad():
        for i in iterator:
            start = max(0, i + 1 - window_length)
            context = input_ids[:, start : i + 1]
            target = input_ids[:, i + 1 : i + 2]
            outputs = model(input_ids=context, use_cache=False)
            logits = outputs.logits[:, -1, :]
            step_loss = loss_fct(logits, target.view(-1))
            step_loss_value = float(step_loss.item())
            losses.append(step_loss_value)
            if step_callback is not None:
                step_callback(i, step_loss_value)
            if log_interval > 0 and (i + 1) % log_interval == 0:
                print(f"step {i + 1}: loss={step_loss_value:.6f}")
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="/data/pretrained_models/Llama-2-7b-hf")
    parser.add_argument("--input", default="data/pg19_20k.txt")
    parser.add_argument("--window-length", type=int, default=2048)
    parser.add_argument("--num-sink-tokens", type=int, default=4)
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument(
        "--strategy",
        default="dense",
        choices=["dense", "window", "sink", "recompute"],
    )
    parser.add_argument("--wandb-project", default="streamingllm-exp")
    parser.add_argument("--wandb-run", default="exp1_ppl")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--log-interval", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    print(f"Model loaded in {time.time() - start:.2f}s")

    strategies = {
        "dense": DenseCache(),
        "window": WindowedCache(max_len=args.window_length),
        "sink": SinkCacheStrategy(
            window_length=args.window_length, num_sink_tokens=args.num_sink_tokens
        ),
    }

    if args.wandb_mode != "disabled":
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                "model_name": args.model_name,
                "input": args.input,
                "window_length": args.window_length,
                "num_sink_tokens": args.num_sink_tokens,
                "smooth_window": args.smooth_window,
                "max_tokens": args.max_tokens,
                "strategy": args.strategy,
            },
            mode=args.wandb_mode,
        )

    step_callback = None
    if args.wandb_mode != "disabled":
        window = max(1, args.smooth_window)
        recent = deque(maxlen=window)

        def step_callback(step, loss_value):
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

    if args.strategy == "recompute":
        losses = run_recompute(
            model,
            input_ids,
            window_length=args.window_length,
            log_interval=args.log_interval,
            step_callback=step_callback,
        )
    else:
        strategy = strategies[args.strategy]
        kv_cache = strategy.init_cache()
        losses = compute_streaming_loss(
            model,
            input_ids,
            kv_cache=kv_cache,
            cache_strategy=strategy,
            log_interval=args.log_interval,
            progress_desc=args.strategy,
            step_callback=step_callback,
        )

    losses = [float(x) for x in losses]
    smoothed = sliding_window_mean(losses, args.smooth_window)
    ppl = losses_to_ppl(smoothed)

    if args.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
