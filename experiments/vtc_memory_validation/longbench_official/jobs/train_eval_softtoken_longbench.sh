#!/usr/bin/env bash
set -euo pipefail

# Train a soft-token compressor, generate LongBench predictions, and score them
# with official_eval/eval.py.
#
# Required environment:
#   DECODER      Model registry name or filesystem path.
#   CONFIG       softtoken/train.py config path.
#   FACTOR       Compression factor used by pred_softtoken.py.
#   DATA         Training data path for softtoken/train.py.
#   CKPT         Checkpoint output path.
#   RUN_NAME     Name under longbench_official/official_eval/pred/.
#
# Optional environment:
#   RESULTS_DIR  Directory to copy result_${RUN_NAME}.json into.
#   DATASETS     Comma-separated LongBench datasets.
#   LIMIT        Max examples per LongBench dataset for smoke tests.
#   ENC_WINDOW   Encoder window for soft-token LongBench prediction.
#   CUDA_VISIBLE_DEVICES

cd "$(dirname "$0")/../.."

: "${DECODER:?Set DECODER to a model name or path}"
: "${CONFIG:?Set CONFIG}"
: "${FACTOR:?Set FACTOR}"
: "${DATA:?Set DATA}"
: "${CKPT:?Set CKPT}"
: "${RUN_NAME:?Set RUN_NAME}"

RESULTS_DIR="${RESULTS_DIR:-/shared/public/sharing/vtc_memory/results}"
DATASETS="${DATASETS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique}"
LIMIT="${LIMIT:-0}"
ENC_WINDOW="${ENC_WINDOW:-512}"

python -u softtoken/train.py \
  --config "${CONFIG}" \
  --factor "${FACTOR}" \
  --decoder "${DECODER}" \
  --data "${DATA}" \
  --save "${CKPT}"

python -u longbench_official/pred_softtoken.py \
  --datasets "${DATASETS}" \
  --condition softtoken \
  --run_name "${RUN_NAME}" \
  --decoder "${DECODER}" \
  --softtoken_ckpt "${CKPT}" \
  --factor "${FACTOR}" \
  --enc_window "${ENC_WINDOW}" \
  --limit "${LIMIT}"

(
  cd longbench_official/official_eval
  python -u eval.py --model "${RUN_NAME}"
)

mkdir -p "${RESULTS_DIR}"
cp "longbench_official/official_eval/pred/${RUN_NAME}/result.json" \
  "${RESULTS_DIR}/result_${RUN_NAME}.json"
cat "${RESULTS_DIR}/result_${RUN_NAME}.json"
