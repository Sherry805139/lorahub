#!/bin/bash

# 使用 LoraHub 融合后的全局 LoRA 模型，在 lorahub/infer.jsonl 上用 swift infer 做推理，
# 然后用 test_swift.py 计算 step / episode-level 准确率。
# 用法示例：bash test_merged_lora.sh [GPU_ID]

GPU_ID=${1:-0}

MERGED_DIR="./output/lorahub_global_lora_merged"
VAL_DATASET="./infer.jsonl"

echo "[INFO] Using GPU: $GPU_ID"
echo "[INFO] Merged model dir: $MERGED_DIR"
echo "[INFO] Val dataset: $VAL_DATASET"

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=602112

# 1）用 swift infer 对融合后的全参数模型做推理
swift infer \
  --ckpt_dir "$MERGED_DIR" \
  --val_dataset "$VAL_DATASET" \
  --model_type qwen2-vl-2b-instruct \
  --model_id_or_path "$MERGED_DIR" \
  --sft_type full

if [ $? -ne 0 ]; then
  echo "[ERROR] swift infer failed."
  exit 1
fi

# 2）找到最新的推理结果 jsonl，并用 test_swift.py 计算准确率
JSONL_DIR="$MERGED_DIR/infer_result"
jsonl_files=$(find "$JSONL_DIR" -maxdepth 1 -type f -name "*.jsonl" 2>/dev/null | sort)
latest_jsonl=$(echo "$jsonl_files" | tail -n 1)

if [ -z "$latest_jsonl" ]; then
  echo "[ERROR] No inference jsonl found in $JSONL_DIR"
  exit 1
fi

echo "[INFO] Evaluating merged model with test_swift.py on: $latest_jsonl"
python test_swift.py --data_path "$latest_jsonl"


