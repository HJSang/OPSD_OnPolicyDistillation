#!/usr/bin/env bash
set -euo pipefail

# Generate raw LongBench predictions and score them with official_eval/eval.py.
#
# Required environment:
#   DECODER      Model registry name or filesystem path.
#   RUN_NAME     Name under longbench_official/official_eval/pred/.
#
# Optional environment:
#   RESULTS_DIR  Directory to copy result_${RUN_NAME}.json into.
#   DATASETS     Comma-separated LongBench datasets.
#   CUDA_VISIBLE_DEVICES

cd "$(dirname "$0")/../.."

: "${DECODER:?Set DECODER to a model name or path}"
: "${RUN_NAME:?Set RUN_NAME}"

RESULTS_DIR="${RESULTS_DIR:-/shared/public/sharing/vtc_memory/results}"
DATASETS="${DATASETS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique}"

python -u longbench_official/pred_softtoken.py \
  --datasets "${DATASETS}" \
  --condition raw \
  --run_name "${RUN_NAME}" \
  --decoder "${DECODER}"

(
  cd longbench_official/official_eval
  python -u eval.py --model "${RUN_NAME}"
)

mkdir -p "${RESULTS_DIR}"
cp "longbench_official/official_eval/pred/${RUN_NAME}/result.json" \
  "${RESULTS_DIR}/result_${RUN_NAME}.json"
cat "${RESULTS_DIR}/result_${RUN_NAME}.json"
