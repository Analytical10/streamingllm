#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-both}"

MODEL_NAME="${MODEL_NAME:-/data/pretrained_models/Llama-2-7b-hf}"
INPUT_FILE="${INPUT_FILE:-data/pg19_books_65133.txt}"
TOTAL_CACHE_SIZE="${TOTAL_CACHE_SIZE:-4096}"
SINK_SIZES="${SINK_SIZES:-0 1 2 4 8}"
WANDB_PROJECT="${WANDB_PROJECT:-streamingllm-exp5}"
WANDB_MODE="${WANDB_MODE:-online}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-100}"

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Error: dataset ${INPUT_FILE} not found."
  exit 1
fi

run_exp5() {
  local output_dir="$1"
  local max_tokens="$2"

  python scripts/run_exp5_sink_ablation.py \
    --model-name "${MODEL_NAME}" \
    --input "${INPUT_FILE}" \
    --total-cache-size "${TOTAL_CACHE_SIZE}" \
    --sink-sizes ${SINK_SIZES} \
    --max-tokens "${max_tokens}" \
    --smooth-window "${SMOOTH_WINDOW}" \
    --output-dir "${output_dir}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-mode "${WANDB_MODE}" \
    --log-interval "${LOG_INTERVAL}"
}

case "${MODE}" in
  smoke)
    echo "Running Exp5 smoke test..."
    run_exp5 "outputs/exp5/smoke" 512
    ;;
  full)
    echo "Running Exp5 full ablation..."
    run_exp5 "outputs/exp5" 0
    ;;
  both)
    echo "Running Exp5 smoke test before full ablation..."
    run_exp5 "outputs/exp5/smoke" 512
    echo "Smoke test complete. Running Exp5 full ablation..."
    run_exp5 "outputs/exp5" 0
    ;;
  *)
    echo "Usage: bash scripts/run_exp5.sh [smoke|full|both]"
    exit 1
    ;;
esac

echo "Exp5 ${MODE} run completed successfully."
