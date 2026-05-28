#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="/data/pretrained_models/Llama-2-7b-hf"
INPUT_PATH="data/pg19_20k.txt"
WINDOW_LENGTH=2048
NUM_SINK_TOKENS=4
SMOOTH_WINDOW=100
MAX_TOKENS=20000
##"The strategy to compute perplexity. Choices are: dense, window, sink, recompute.",
STRATEGY="dense"
WANDB_PROJECT="streamingllm-exp"
WANDB_RUN="exp1_${STRATEGY}"
WANDB_MODE="online"
LOG_INTERVAL=1000

python scripts/run_exp1_ppl.py \
  --model-name "${MODEL_NAME}" \
  --input "${INPUT_PATH}" \
  --window-length "${WINDOW_LENGTH}" \
  --num-sink-tokens "${NUM_SINK_TOKENS}" \
  --smooth-window "${SMOOTH_WINDOW}" \
  --max-tokens "${MAX_TOKENS}" \
  --strategy "${STRATEGY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run "${WANDB_RUN}" \
  --wandb-mode "${WANDB_MODE}" \
  --log-interval "${LOG_INTERVAL}"
