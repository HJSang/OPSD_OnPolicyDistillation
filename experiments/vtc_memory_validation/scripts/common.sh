#!/usr/bin/env bash

if [[ -n "${VTC_COMMON_LOADED:-}" ]]; then
  return 0
fi
export VTC_COMMON_LOADED=1

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VTC_EXPERIMENT_DIR="${VTC_EXPERIMENT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export VTC_REPO_ROOT="${VTC_REPO_ROOT:-$(cd -- "${VTC_EXPERIMENT_DIR}/../.." && pwd)}"
export VTC_DATA_DIR="${VTC_DATA_DIR:-${VTC_EXPERIMENT_DIR}/data}"
export VTC_RESULTS_DIR="${VTC_RESULTS_DIR:-${VTC_EXPERIMENT_DIR}/results}"
export VTC_CHECKPOINT_DIR="${VTC_CHECKPOINT_DIR:-${VTC_EXPERIMENT_DIR}/checkpoints}"
export PYTHON="${PYTHON:-python3}"

mkdir -p "${VTC_DATA_DIR}" "${VTC_RESULTS_DIR}" "${VTC_CHECKPOINT_DIR}"
cd "${VTC_EXPERIMENT_DIR}"

ensure_memory_data() {
  local missing=0
  local name
  local bundled_ultrachat="${VTC_EXPERIMENT_DIR}/longmemeval_evaluation_training_data/ultrachat_train.json"
  for name in "$@"; do
    if [[ "${name}" == "ultrachat_train.json" &&
          ! -s "${VTC_DATA_DIR}/${name}" &&
          -s "${bundled_ultrachat}" ]]; then
      cp "${bundled_ultrachat}" "${VTC_DATA_DIR}/${name}"
    fi
  done
  for name in "$@"; do
    if [[ ! -s "${VTC_DATA_DIR}/${name}" ]]; then
      missing=1
    fi
  done
  if (( missing )); then
    "${PYTHON}" prepare_data.py --data_dir "${VTC_DATA_DIR}"
  fi
  for name in "$@"; do
    if [[ ! -s "${VTC_DATA_DIR}/${name}" ]]; then
      echo "Required data file was not prepared: ${VTC_DATA_DIR}/${name}" >&2
      return 1
    fi
  done
}

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Required file not found: $1" >&2
    return 1
  fi
}

enable_flag() {
  [[ "${1,,}" == "1" || "${1,,}" == "true" || "${1,,}" == "yes" ]]
}
