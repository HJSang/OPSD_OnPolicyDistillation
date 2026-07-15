#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_memory_data ultrachat_train.json longmemeval_oracle.json

DECODER="${DECODER:-qwen2.5-7b}"
FACTORS="${FACTORS:-4 8 16}"
STEPS="${STEPS:-1000}"
N_CHUNKS="${N_CHUNKS:-400}"
MAX_LEN="${MAX_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
ENC_LAYERS="${ENC_LAYERS:-2}"
POOL="${POOL:-mean}"
FKL_WEIGHT="${FKL_WEIGHT:-1.0}"
TRAIN_ENCODER="${TRAIN_ENCODER:-1}"
LIMIT="${LIMIT:-500}"
SEED="${SEED:-0}"
TRAIN_DATASET="${TRAIN_DATASET:-ultrachat}"
TRAIN_DATA="${TRAIN_DATA:-${VTC_DATA_DIR}/ultrachat_train.json}"
EVAL_DATASET="${EVAL_DATASET:-longmemeval}"
EVAL_DATA="${EVAL_DATA:-${VTC_DATA_DIR}/longmemeval_oracle.json}"

for factor in ${FACTORS}; do
  ckpt="${VTC_CHECKPOINT_DIR}/softtoken_simple_f${factor}.pt"
  out="${VTC_RESULTS_DIR}/results_softtoken_simple_f${factor}.json"
  train_args=(
    --decoder "${DECODER}"
    --dataset "${TRAIN_DATASET}"
    --data "${TRAIN_DATA}"
    --factor "${factor}"
    --mode simple
    --pool "${POOL}"
    --enc_layers "${ENC_LAYERS}"
    --max_len "${MAX_LEN}"
    --n_chunks "${N_CHUNKS}"
    --batch_size "${BATCH_SIZE}"
    --steps "${STEPS}"
    --lr "${LR}"
    --fkl_weight "${FKL_WEIGHT}"
    --save "${ckpt}"
  )
  if enable_flag "${TRAIN_ENCODER}"; then
    train_args+=(--train_encoder)
  fi

  "${PYTHON}" -u softtoken/train.py "${train_args[@]}" "$@"
  "${PYTHON}" -u softtoken/eval_softtoken.py \
    --ckpt "${ckpt}" \
    --dataset "${EVAL_DATASET}" \
    --data "${EVAL_DATA}" \
    --limit "${LIMIT}" \
    --shuffle \
    --seed "${SEED}" \
    --out "${out}"
done
