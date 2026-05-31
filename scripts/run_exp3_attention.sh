#!/bin/bash
set -e

# Change directory to the workspace root
cd "$(dirname "$0")/.."

mkdir -p outputs

echo "Running Exp3: Llama-2-7B Attention Analysis"
python scripts/run_exp3_attention.py \
    --model-name /data/pretrained_models/Llama-2-7b-hf \
    --input data/pg19_20k.txt \
    --output-dir outputs

echo "Done!"
