#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DECODER="${DECODER:-qwen2.5-7b}"
RUN_NAME="${RUN_NAME:-longbench_raw}"
LIMIT="${LIMIT:-0}"
DATASETS="${DATASETS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique}"

"${PYTHON}" -u longbench_official/pred_softtoken.py \
  --datasets "${DATASETS}" \
  --condition raw \
  --run_name "${RUN_NAME}" \
  --decoder "${DECODER}" \
  --limit "${LIMIT}" \
  "$@"

(
  cd longbench_official/official_eval
  "${PYTHON}" -u eval.py --model "${RUN_NAME}"
)

cp "longbench_official/official_eval/pred/${RUN_NAME}/result.json" \
  "${VTC_RESULTS_DIR}/result_${RUN_NAME}.json"
cat "${VTC_RESULTS_DIR}/result_${RUN_NAME}.json"
