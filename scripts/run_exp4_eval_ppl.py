import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from streamingllm_experiment.cache_utils import SinkCacheStrategy, WindowedCache
from streamingllm_experiment.eval_utils import compute_streaming_loss, losses_to_ppl, sliding_window_mean
from streamingllm_experiment.logging_utils import save_csv
from streamingllm_experiment.paths import OUTPUTS_DIR
from streamingllm_experiment.tokenization import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input", default="data/pg19_20k.txt")
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "exp4_eval"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_name)
    text = Path(args.input).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto"
    )

    configs = [
        ("0_1024", WindowedCache(max_len=args.window_length)),
        ("1_1023", SinkCacheStrategy(window_length=args.window_length, num_sink_tokens=1)),
        ("4_1020", SinkCacheStrategy(window_length=args.window_length, num_sink_tokens=4)),
    ]

    for name, strategy in configs:
        kv_cache = strategy.init_cache()
        losses = compute_streaming_loss(model, input_ids, kv_cache=kv_cache)
        smoothed = sliding_window_mean(losses, args.smooth_window)
        ppl = losses_to_ppl(smoothed)
        rows = [
            {"step": i, "loss": losses[i], "loss_smooth": smoothed[i], "ppl": ppl[i]}
            for i in range(len(losses))
        ]
        save_csv(Path(args.output_dir) / f"{name}.csv", rows)


if __name__ == "__main__":
    main()
