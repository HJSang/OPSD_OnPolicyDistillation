#!/usr/bin/env bash
set -euo pipefail

docker_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "${docker_dir}/.." && pwd)"
repo_root="$(cd -- "${experiment_dir}/../.." && pwd)"

image="${VTC_DOCKER_IMAGE:-vtc-memory-validation:cu129}"
hf_home="${VTC_HF_HOME:-${HOME}/.cache/huggingface}"
runs_dir="${VTC_RUNS_DIR:-${experiment_dir}/.docker-runs}"

mkdir -p "${hf_home}" "${runs_dir}"/{data,results,checkpoints,logs}

tty_args=()
if [[ -t 0 && -t 1 ]]; then
  tty_args=(-it)
fi

env_args=()
for name in \
  CUDA_VISIBLE_DEVICES \
  HF_TOKEN \
  JUDGE_MODEL \
  JUDGE_QUANTIZATION \
  JUDGE_REVISION \
  JUDGE_TP \
  VTC_MODEL_QWEN2_5_7B \
  VTC_MODEL_QWEN3_4B \
  VTC_MODEL_QWEN3_5_4B \
  VTC_MODEL_QWEN3_8B \
  VTC_MODEL_QWEN2_5_VL_7B \
  VTC_MODEL_DEEPSEEK_OCR
do
  if [[ -n "${!name:-}" ]]; then
    env_args+=(--env "${name}")
  fi
done

if (( $# == 0 )); then
  set -- bash
fi

exec docker run --rm "${tty_args[@]}" \
  --gpus all \
  --ipc host \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp/vtc-home \
  --env USER=vtc \
  --env LOGNAME=vtc \
  --env TORCHINDUCTOR_CACHE_DIR=/tmp/vtc-home/.cache/torchinductor \
  "${env_args[@]}" \
  --volume "${repo_root}:/workspace" \
  --volume "${hf_home}:/cache/huggingface" \
  --volume "${runs_dir}:/runs" \
  --workdir /workspace/experiments/vtc_memory_validation \
  "${image}" \
  "$@"
