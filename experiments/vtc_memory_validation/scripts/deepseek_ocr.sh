#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_memory_data longmemeval_oracle.json

if [[ -x "${VTC_EXPERIMENT_DIR}/.venv-deepseek-ocr/bin/python" ]]; then
  default_dsocr_python="${VTC_EXPERIMENT_DIR}/.venv-deepseek-ocr/bin/python"
else
  default_dsocr_python="${PYTHON}"
fi
DSOCR_PYTHON="${DSOCR_PYTHON:-${default_dsocr_python}}"
DSOCR_MODEL="${DSOCR_MODEL:-deepseek-ocr}"
TEXT_MODEL="${TEXT_MODEL:-qwen2.5-7b}"
BASE_SIZES="${BASE_SIZES:-1024 640 512}"
LIMIT="${LIMIT:-100}"
SEED="${SEED:-0}"
MODE="${MODE:-simple}"
FONT_SIZE="${FONT_SIZE:-16}"
RENDER_SIZE="${RENDER_SIZE:-1024}"

for base_size in ${BASE_SIZES}; do
  cache="${VTC_RESULTS_DIR}/dsocr_cache_longmemeval_b${base_size}.json"
  out="${VTC_RESULTS_DIR}/results_longmemeval_dsocr_b${base_size}.json"
  "${DSOCR_PYTHON}" -u run_dsocr_reconstruct.py \
    --dataset longmemeval \
    --data_path "${VTC_DATA_DIR}/longmemeval_oracle.json" \
    --limit "${LIMIT}" \
    --shuffle \
    --seed "${SEED}" \
    --mode "${MODE}" \
    --base_size "${base_size}" \
    --render_size "${RENDER_SIZE}" \
    --font_size "${FONT_SIZE}" \
    --model_path "${DSOCR_MODEL}" \
    --out "${cache}" \
    "$@"

  "${PYTHON}" -u run_validation.py \
    --dataset longmemeval \
    --data_path "${VTC_DATA_DIR}/longmemeval_oracle.json" \
    --conditions dsocr \
    --dsocr_cache "${cache}" \
    --limit "${LIMIT}" \
    --shuffle \
    --seed "${SEED}" \
    --text_model "${TEXT_MODEL}" \
    --out "${out}"
done
