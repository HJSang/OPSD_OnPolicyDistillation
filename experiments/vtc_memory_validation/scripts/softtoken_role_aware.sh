#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_memory_data ultrachat_train.json longmemeval_oracle.json

DECODER="${DECODER:-qwen2.5-7b}"
USER_FACTOR="${USER_FACTOR:-1}"
ASSISTANT_FACTORS="${ASSISTANT_FACTORS:-8 16 32}"
BASE_FACTOR="${BASE_FACTOR:-8}"
STEPS="${STEPS:-1000}"
N_CHUNKS="${N_CHUNKS:-400}"
MAX_LEN="${MAX_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
ENC_LAYERS="${ENC_LAYERS:-2}"
POOL="${POOL:-mean}"
FKL_WEIGHT="${FKL_WEIGHT:-1.0}"
TRAIN_ENCODER="${TRAIN_ENCODER:-0}"
LIMIT="${LIMIT:-500}"
SEED="${SEED:-0}"
TRAIN_DATASET="${TRAIN_DATASET:-ultrachat}"
TRAIN_DATA="${TRAIN_DATA:-${VTC_DATA_DIR}/ultrachat_train.json}"
EVAL_DATASET="${EVAL_DATASET:-longmemeval}"
EVAL_DATA="${EVAL_DATA:-${VTC_DATA_DIR}/longmemeval_oracle.json}"

for assistant_factor in ${ASSISTANT_FACTORS}; do
  name="softtoken_u${USER_FACTOR}_a${assistant_factor}"
  ckpt="${VTC_CHECKPOINT_DIR}/${name}.pt"
  out="${VTC_RESULTS_DIR}/results_${name}.json"
  train_args=(
    --decoder "${DECODER}"
    --dataset "${TRAIN_DATASET}"
    --data "${TRAIN_DATA}"
    --factor "${BASE_FACTOR}"
    --mode full
    --user_factor "${USER_FACTOR}"
    --assistant_factor "${assistant_factor}"
    --pool "${POOL}"
    --enc_layers "${ENC_LAYERS}"
    --max_len "${MAX_LEN}"
    --n_chunks "${N_CHUNKS}"
    --batch_size "${BATCH_SIZE}"
    --steps "${STEPS}"
    --lr "${LR}"
    --fkl_weight "${FKL_WEIGHT}"
    --seed "${SEED}"
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
    --skip_judge \
    --out "${out}"
done
