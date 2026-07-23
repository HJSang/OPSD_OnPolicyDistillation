#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DECODER="${DECODER:-qwen2.5-7b}"
TRAIN_SOURCE="${TRAIN_SOURCE:-sourceqa}"
FACTOR="${FACTOR:-8}"
STEPS="${STEPS:-1000}"
QA_ACCUM="${QA_ACCUM:-8}"
MAX_LEN="${MAX_LEN:-2048}"
LIMIT="${LIMIT:-0}"
ENC_WINDOW="${ENC_WINDOW:-512}"
DATASETS="${DATASETS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique}"

case "${TRAIN_SOURCE}" in
  sourceqa)
    TRAIN_DATA="${TRAIN_DATA:-${VTC_EXPERIMENT_DIR}/longbench_evaluation_training_data/longbench_sourceqa_train_balanced.json}"
    ;;
  longmemeval)
    TRAIN_DATA="${TRAIN_DATA:-${VTC_EXPERIMENT_DIR}/longbench_evaluation_training_data/longmemeval_qa_train.json}"
    ;;
  *)
    TRAIN_DATA="${TRAIN_DATA:-${TRAIN_SOURCE}}"
    ;;
esac
require_file "${TRAIN_DATA}"

run_name="${RUN_NAME:-longbench_${TRAIN_SOURCE}_f${FACTOR}}"
ckpt="${CKPT:-${VTC_CHECKPOINT_DIR}/${run_name}.pt}"

"${PYTHON}" -u softtoken/train.py \
  --decoder "${DECODER}" \
  --dataset longmemeval \
  --data "${TRAIN_DATA}" \
  --factor "${FACTOR}" \
  --mode simple \
  --pool mean \
  --enc_layers 2 \
  --qa_train \
  --no_abstain \
  --qa_accum "${QA_ACCUM}" \
  --steps "${STEPS}" \
  --n_chunks 400 \
  --max_len "${MAX_LEN}" \
  --batch_size 1 \
  --lr 1e-4 \
  --save "${ckpt}" \
  "$@"

"${PYTHON}" -u longbench_official/pred_softtoken.py \
  --datasets "${DATASETS}" \
  --condition softtoken \
  --run_name "${run_name}" \
  --decoder "${DECODER}" \
  --softtoken_ckpt "${ckpt}" \
  --factor "${FACTOR}" \
  --enc_window "${ENC_WINDOW}" \
  --limit "${LIMIT}"

(
  cd longbench_official/official_eval
  "${PYTHON}" -u eval.py --model "${run_name}"
)

cp "longbench_official/official_eval/pred/${run_name}/result.json" \
  "${VTC_RESULTS_DIR}/result_${run_name}.json"
cat "${VTC_RESULTS_DIR}/result_${run_name}.json"
