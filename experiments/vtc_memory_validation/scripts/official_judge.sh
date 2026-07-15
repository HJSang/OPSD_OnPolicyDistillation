#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_memory_data longmemeval_oracle.json

MODEL="${JUDGE_MODEL:-meta-llama/Meta-Llama-3.1-70B-Instruct}"
REVISION="${JUDGE_REVISION:-1605565b47bb9346c5515c34102e054115b4f98b}"
TP="${JUDGE_TP:-2}"
QUANTIZATION="${JUDGE_QUANTIZATION:-fp8}"

if (( $# == 0 )); then
  echo "Usage: $0 RESULT_JSON [RESULT_JSON ...]" >&2
  exit 2
fi

for input in "$@"; do
  output="${input%.json}_official_judge.json"
  "${PYTHON}" -u official_longmemeval_judge.py \
    --input "${input}" \
    --data "${VTC_DATA_DIR}/longmemeval_oracle.json" \
    --output "${output}" \
    --model "${MODEL}" \
    --revision "${REVISION}" \
    --tensor-parallel-size "${TP}" \
    --quantization "${QUANTIZATION}"
done
