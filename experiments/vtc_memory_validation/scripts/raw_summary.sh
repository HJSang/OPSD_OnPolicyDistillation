#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_memory_data longmemeval_oracle.json

LIMIT="${LIMIT:-100}"
SEED="${SEED:-0}"
TEXT_MODEL="${TEXT_MODEL:-qwen2.5-7b}"
CONDITIONS="${CONDITIONS:-raw,summary}"
OUT="${OUT:-${VTC_RESULTS_DIR}/results_longmemeval_raw_summary_${LIMIT}.json}"

"${PYTHON}" -u run_validation.py \
  --dataset longmemeval \
  --data_path "${VTC_DATA_DIR}/longmemeval_oracle.json" \
  --conditions "${CONDITIONS}" \
  --limit "${LIMIT}" \
  --shuffle \
  --seed "${SEED}" \
  --text_model "${TEXT_MODEL}" \
  --out "${OUT}" \
  "$@"
