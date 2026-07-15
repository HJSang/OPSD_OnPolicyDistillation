#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "${script_dir}/.." && pwd)"
runner="${experiment_dir}/docker/run.sh"

runs_dir="${VTC_RUNS_DIR:-${experiment_dir}/.docker-runs/main-table}"
qwen_revision="a09a35458c702b33eeacc393d103063234e8bc28"
dsocr_revision="9f30c71f441d010e5429c532364a86705536c53a"
llama_revision="1605565b47bb9346c5515c34102e054115b4f98b"
qwen_path="/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/${qwen_revision}"
dsocr_path="/cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/${dsocr_revision}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for the gated Llama judge model." >&2
  exit 2
fi

mkdir -p "${runs_dir}"/{data,results,checkpoints,logs}
export VTC_RUNS_DIR="${runs_dir}"
export VTC_MODEL_QWEN2_5_7B="${qwen_path}"
export VTC_MODEL_DEEPSEEK_OCR="${dsocr_path}"

{
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'git_commit=%s\n' "$(git -C "${experiment_dir}" rev-parse HEAD)"
  printf 'git_dirty=\n'
  git -C "${experiment_dir}" status --short
  printf 'qwen_revision=%s\n' "${qwen_revision}"
  printf 'deepseek_ocr_revision=%s\n' "${dsocr_revision}"
  printf 'llama_judge_revision=%s\n' "${llama_revision}"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader
} >"${runs_dir}/logs/run-manifest.log"

run_container() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu}" "${runner}" "$@"
}

echo "[reproduce] runs: ${runs_dir}"
echo "[reproduce] verifying image and downloading pinned models"
run_container 0 python docker/verify_environment.py --require-gpu \
  >"${runs_dir}/logs/environment.log" 2>&1
run_container 0 hf download Qwen/Qwen2.5-7B-Instruct \
  --revision "${qwen_revision}" \
  >"${runs_dir}/logs/download-qwen.log" 2>&1
run_container 0 hf download deepseek-ai/DeepSeek-OCR \
  --revision "${dsocr_revision}" \
  >"${runs_dir}/logs/download-deepseek-ocr.log" 2>&1
run_container 0 hf download meta-llama/Meta-Llama-3.1-70B-Instruct \
  --revision "${llama_revision}" \
  --exclude "original/*" \
  >"${runs_dir}/logs/download-llama-judge.log" 2>&1
run_container 0 python prepare_data.py --data_dir /runs/data \
  >"${runs_dir}/logs/prepare-data.log" 2>&1

pids=()
names=()
launch() {
  local name="$1"
  local gpu="$2"
  local command="$3"
  echo "[reproduce] launch ${name} on GPU ${gpu}"
  run_container "${gpu}" bash -lc "${command}" \
    >"${runs_dir}/logs/${name}.log" 2>&1 &
  pids+=("$!")
  names+=("${name}")
}

launch uniform-f4 0 "FACTORS=4 LIMIT=500 bash scripts/softtoken_simple.sh"
launch uniform-f8 1 "FACTORS=8 LIMIT=500 bash scripts/softtoken_simple.sh"
launch uniform-f16 2 "FACTORS=16 LIMIT=500 bash scripts/softtoken_simple.sh"
launch role-a8 3 "ASSISTANT_FACTORS=8 LIMIT=500 bash scripts/softtoken_role_aware.sh"
launch role-a16 4 "ASSISTANT_FACTORS=16 LIMIT=500 bash scripts/softtoken_role_aware.sh"
launch role-a32 5 "ASSISTANT_FACTORS=32 LIMIT=500 bash scripts/softtoken_role_aware.sh"
launch raw-summary 6 \
  "LIMIT=500 OUT=/runs/results/results_longmemeval_raw_summary_500.json bash scripts/raw_summary.sh"
launch deepseek-ocr 7 \
  "LIMIT=500 BASE_SIZES='1024 640 512' bash scripts/deepseek_ocr.sh"

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[reproduce] completed ${names[$i]}"
  else
    echo "[reproduce] FAILED ${names[$i]} (see logs/${names[$i]}.log)" >&2
    failed=1
  fi
done
if (( failed )); then
  exit 1
fi

echo "[reproduce] scoring all rows with one Llama judge load"
run_container "0,1" python official_longmemeval_judge.py \
  --data /runs/data/longmemeval_oracle.json \
  --revision "${llama_revision}" \
  --job /runs/results/results_longmemeval_raw_summary_500.json:raw_pred \
  --job /runs/results/results_longmemeval_raw_summary_500.json:summary_pred \
  --job /runs/results/results_softtoken_simple_f4.json:pred \
  --job /runs/results/results_softtoken_simple_f8.json:pred \
  --job /runs/results/results_softtoken_simple_f16.json:pred \
  --job /runs/results/results_softtoken_u1_a8.json:pred \
  --job /runs/results/results_softtoken_u1_a16.json:pred \
  --job /runs/results/results_softtoken_u1_a32.json:pred \
  --job /runs/results/results_longmemeval_dsocr_b1024.json:dsocr_pred \
  --job /runs/results/results_longmemeval_dsocr_b640.json:dsocr_pred \
  --job /runs/results/results_longmemeval_dsocr_b512.json:dsocr_pred \
  >"${runs_dir}/logs/official-judge.log" 2>&1

echo "[reproduce] complete: ${runs_dir}"
