#!/bin/bash
set -ex

MODEL_NAME="/data/pretrained_models/Llama-2-7b-hf"
INPUT_FILE="data/pg19_books_65133.txt"

# 1. 确保数据存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Dataset $INPUT_FILE not found."
    exit 1
fi

# 可选：如果希望本地调试速度更快，加上 --max-tokens 5000 进行截断测试
MAX_TOKENS=0 # 设置为 0 跑全量，设置为 5000 跑本地快速测试

#2. Baseline (Window 1024)
echo "Running Baseline (Window 1024)..."
python scripts/run_exp4_attention_sink.py \
    --model-name $MODEL_NAME \
    --input $INPUT_FILE \
    --strategy window \
    --window-length 1024 \
    --max-tokens $MAX_TOKENS \
    --wandb-run "exp4_window_1024"

# 3. StreamingLLM (Sink 4+1020)
echo "Running StreamingLLM (Sink 4+1020)..."
python scripts/run_exp4_attention_sink.py \
    --model-name $MODEL_NAME \
    --input $INPUT_FILE \
    --strategy sink \
    --window-length 1024 \
    --num-sink-tokens 4 \
    --max-tokens $MAX_TOKENS \
    --wandb-run "exp4_sink_4_1020"

# 4. Newline Sink (4"\n"+1020)
echo "Running Newline Sink (4\"\n\"+1020)..."
python scripts/run_exp4_attention_sink.py \
    --model-name $MODEL_NAME \
    --input $INPUT_FILE \
    --strategy sink \
    --window-length 1024 \
    --num-sink-tokens 4 \
    --max-tokens $MAX_TOKENS \
    --replace-sink-newline \
    --wandb-run "exp4_sink_newline_4_1020"

echo "All Exp4 experiments completed successfully!"