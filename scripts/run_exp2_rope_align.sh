#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="/data/pretrained_models/Llama-2-7b-hf"
INPUT_PATH="data/pg19_20k.txt"
WINDOW_LENGTH=2048
SMOOTH_WINDOW=100
MAX_TOKENS=20000

# The alignment mode: "aligned" or "misaligned"
ALIGNMENT="aligned"

WANDB_PROJECT="streamingllm-exp"
WANDB_RUN="exp2_${ALIGNMENT}"
WANDB_MODE="online"
LOG_INTERVAL=1000

python scripts/run_exp2_rope_align.py \
  --model-name "${MODEL_NAME}" \
  --input "${INPUT_PATH}" \
  --window-length "${WINDOW_LENGTH}" \
  --smooth-window "${SMOOTH_WINDOW}" \
  --max-tokens "${MAX_TOKENS}" \
  --alignment "${ALIGNMENT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run "${WANDB_RUN}" \
  --wandb-mode "${WANDB_MODE}" \
  --log-interval "${LOG_INTERVAL}"
