#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_memory_data ultrachat_train.json

DECODER="${DECODER:-qwen3-8b}"
TRAIN_DATA="${TRAIN_DATA:-${VTC_DATA_DIR}/ultrachat_train.json}"
FACTOR="${FACTOR:-3}"
STEPS="${STEPS:-1000}"
RUN_NAME="${RUN_NAME:-long_context_f${FACTOR}}"

"${PYTHON}" -u softtoken/train.py \
  --decoder "${DECODER}" \
  --dataset ultrachat \
  --data "${TRAIN_DATA}" \
  --factor "${FACTOR}" \
  --mode simple \
  --pool mean \
  --enc_layers 2 \
  --max_len 2048 \
  --n_chunks 400 \
  --long_context \
  --max_mem_tokens 8192 \
  --min_mem_tokens 512 \
  --target_len 256 \
  --enc_window 512 \
  --batch_size 1 \
  --steps "${STEPS}" \
  --lr 1e-4 \
  --fkl_weight 1.0 \
  --save "${VTC_CHECKPOINT_DIR}/${RUN_NAME}.pt" \
  "$@"
