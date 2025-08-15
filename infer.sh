#!/usr/bin/env bash
# Example: infer.sh best_checkpoint.pt

set -e
MODEL_PATH="$1"

if [ -z "${MODEL_PATH}" ]; then
  echo "Usage: ./infer.sh <model_ckpt.pt>"
  exit 1
fi

# 可根据需要修改
TEST_TXT=./data/dummy.txt
AUDIO_ROOT=./data/dummy
BATCH=512                 # 总 batch，DataParallel 会自动切分到各 GPU
OUT_CSV=./quantitative-result.csv

# 指定使用哪几张卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

python inference.py \
    --wav_file   "${TEST_TXT}" \
    --audio_root "${AUDIO_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --batch_size "${BATCH}" \
    --output_file "${OUT_CSV}"
