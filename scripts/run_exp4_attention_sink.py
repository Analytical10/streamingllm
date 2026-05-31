import argparse
import time
from collections import deque
from pathlib import Path

import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from streamingllm_experiment.cache_utils import DenseCache, SinkCacheStrategy, WindowedCache, get_cache_seq_length
from streamingllm_experiment.eval_utils import compute_streaming_loss, losses_to_ppl, sliding_window_mean
from streamingllm_experiment.tokenization import load_tokenizer
from streamingllm_experiment.pos_shift import enable_llama_pos_shift_attention

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="/data/pretrained_models/Llama-2-7b-hf")
    parser.add_argument("--input", default="data/pg19_books_65133.txt")
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--num-sink-tokens", type=int, default=4)
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument(
        "--strategy",
        default="dense",
        choices=["dense", "window", "sink"],
    )
    parser.add_argument("--replace-sink-newline", action="store_true", help="Replace the sink tokens with newline tokens")
    parser.add_argument("--wandb-project", default="streamingllm-exp4")
    parser.add_argument("--wandb-run", default="exp4")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--log-interval", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_name)
    text = Path(args.input).read_text(encoding="utf-8")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    if args.max_tokens > 0:
        input_ids = input_ids[:, : args.max_tokens]

    if args.replace_sink_newline:
        newline_token_ids = tokenizer.encode('\n', add_special_tokens=False)
        newline_id = newline_token_ids[-1]
        print(f"Replacing first {args.num_sink_tokens} tokens with newline (token_id: {newline_id})")
        input_ids[0, :args.num_sink_tokens] = newline_id

    input_ids = input_ids.to(device)
    print(f"Loaded input_ids with seq_len={input_ids.shape[1]}")

    print("Loading model...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    print(f"Model loaded in {time.time() - start:.2f}s")

    enable_llama_pos_shift_attention(model)

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
                "replace_sink_newline": args.replace_sink_newline,
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

    strategy = strategies[args.strategy]
    kv_cache = strategy.init_cache()
    
    def aligned_position_ids_fn(cache):
        cache_len = get_cache_seq_length(cache)
        if cache_len < args.window_length:
            return torch.tensor([[cache_len]], dtype=torch.long, device=device)
        return torch.tensor([[args.window_length - 1]], dtype=torch.long, device=device)

    losses = compute_streaming_loss(
        model,
        input_ids,
        kv_cache=kv_cache,
        custom_position_fn=aligned_position_ids_fn,
        cache_strategy=strategy,
        log_interval=args.log_interval,
        progress_desc=args.strategy,
        step_callback=step_callback,
    )

    losses = [float(x) for x in losses]
    smoothed = sliding_window_mean(losses, args.smooth_window)
    ppl = losses_to_ppl(smoothed)
    overall_avg_loss = sum(losses) / len(losses) if losses else float('inf')
    overall_ppl = torch.tensor(overall_avg_loss).exp().item()
    print(f"Overall Average Loss: {overall_avg_loss:.4f}")
    print(f"Overall Perplexity: {overall_ppl:.2f}")
    # final_ppl = ppl[-1] if ppl else float('inf')
    # print(f"Final Perplexity: {final_ppl:.2f}")

    if args.wandb_mode != "disabled":
        wandb.summary["overall_avg_loss"] = overall_avg_loss
        wandb.summary["overall_ppl"] = overall_ppl
        wandb.finish()


if __name__ == "__main__":
    main()