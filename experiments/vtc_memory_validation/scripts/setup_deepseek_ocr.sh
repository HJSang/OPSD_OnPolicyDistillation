#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DSOCR_VENV="${DSOCR_VENV:-${VTC_EXPERIMENT_DIR}/.venv-deepseek-ocr}"
"${PYTHON}" -m venv --system-site-packages "${DSOCR_VENV}"
"${DSOCR_VENV}/bin/python" -m pip install --upgrade pip
"${DSOCR_VENV}/bin/python" -m pip install \
  -r "${VTC_EXPERIMENT_DIR}/requirements-deepseek-ocr.txt"

echo "DeepSeek-OCR Python: ${DSOCR_VENV}/bin/python"
