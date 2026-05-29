import argparse
import time
from collections import deque
from pathlib import Path

import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from streamingllm_experiment.cache_utils import SinkCacheStrategy
from streamingllm_experiment.eval_utils import compute_streaming_loss, losses_to_ppl, sliding_window_mean
from streamingllm_experiment.position_utils import aligned_position_ids_fn, build_misaligned_position_ids_fn
from streamingllm_experiment.tokenization import load_tokenizer
from streamingllm_experiment.pos_shift import enable_llama_pos_shift_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="/data/pretrained_models/Llama-2-7b-hf")
    parser.add_argument("--input", default="data/pg19_20k.txt")
    parser.add_argument("--window-length", type=int, default=2048)
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument(
        "--alignment",
        default="aligned",
        choices=["aligned", "misaligned"],
    )
    parser.add_argument("--wandb-project", default="streamingllm-exp")
    parser.add_argument("--wandb-run", default="exp2_rope_align")
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

    shift_mode = "kvcache" if args.alignment == "aligned" else "absolute"
    enable_llama_pos_shift_attention(model, shift_mode=shift_mode)

    if args.wandb_mode != "disabled":
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                "model_name": args.model_name,
                "input": args.input,
                "window_length": args.window_length,
                "smooth_window": args.smooth_window,
                "max_tokens": args.max_tokens,
                "alignment": args.alignment,
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

    cache_strategy = SinkCacheStrategy(window_length=args.window_length, num_sink_tokens=4)
    kv_cache = cache_strategy.init_cache()

    # Always provide absolute positions explicitly to test the internal shift logic
    custom_position_fn = build_misaligned_position_ids_fn(device=device)

    losses = compute_streaming_loss(
        model,
        input_ids,
        kv_cache=kv_cache,
        custom_position_fn=custom_position_fn,
        cache_strategy=cache_strategy,
        log_interval=args.log_interval,
        progress_desc=f"{args.alignment}",
        step_callback=step_callback,
    )

    losses_val = [float(x) for x in losses]
    smoothed = sliding_window_mean(losses_val, args.smooth_window)
    ppl = losses_to_ppl(smoothed)

    if args.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
