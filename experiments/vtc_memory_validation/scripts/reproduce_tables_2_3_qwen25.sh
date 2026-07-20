#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "${script_dir}/.." && pwd)"
runner="${experiment_dir}/docker/run.sh"

runs_dir="${VTC_RUNS_DIR:-${experiment_dir}/.docker-runs/tables-2-3-qwen25}"
ocr_cache_dir="${VTC_DSOCR_CACHE_DIR:-}"

base_revision="d149729398750b98c0af14eb82c78cfe92750796"
instruct_revision="a09a35458c702b33eeacc393d103063234e8bc28"
dsocr_revision="9f30c71f441d010e5429c532364a86705536c53a"
judge_revision="1605565b47bb9346c5515c34102e054115b4f98b"

base_path="/cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/${base_revision}"
instruct_path="/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/${instruct_revision}"
dsocr_path="/cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/${dsocr_revision}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for the gated Llama judge model." >&2
  exit 2
fi

mkdir -p "${runs_dir}"/{data,results,checkpoints,logs}
export VTC_RUNS_DIR="${runs_dir}"

run_container() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu}" "${runner}" "$@"
}

{
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'git_commit=%s\n' "$(git -C "${experiment_dir}" rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'qwen_base_revision=%s\n' "${base_revision}"
  printf 'qwen_instruct_revision=%s\n' "${instruct_revision}"
  printf 'deepseek_ocr_revision=%s\n' "${dsocr_revision}"
  printf 'llama_judge_revision=%s\n' "${judge_revision}"
  printf 'seed=0\nprompt_format=chat\n'
  nvidia-smi --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader
} >"${runs_dir}/logs/run-manifest.log"

echo "[tables-2-3] verifying image and downloading pinned models"
run_container 0 python docker/verify_environment.py --require-gpu \
  >"${runs_dir}/logs/environment.log" 2>&1
run_container 0 hf download Qwen/Qwen2.5-7B \
  --revision "${base_revision}" \
  >"${runs_dir}/logs/download-qwen-base.log" 2>&1
run_container 0 hf download Qwen/Qwen2.5-7B-Instruct \
  --revision "${instruct_revision}" \
  >"${runs_dir}/logs/download-qwen-instruct.log" 2>&1
run_container 0 hf download meta-llama/Meta-Llama-3.1-70B-Instruct \
  --revision "${judge_revision}" --exclude "original/*" \
  >"${runs_dir}/logs/download-llama-judge.log" 2>&1
run_container 0 python prepare_data.py --data_dir /runs/data \
  >"${runs_dir}/logs/prepare-data.log" 2>&1

missing_ocr=()
for base_size in 1024 640 512; do
  name="dsocr_cache_longmemeval_b${base_size}.json"
  target="${runs_dir}/results/${name}"
  if [[ -s "${target}" ]]; then
    continue
  fi
  if [[ -n "${ocr_cache_dir}" && -s "${ocr_cache_dir}/${name}" ]]; then
    cp "${ocr_cache_dir}/${name}" "${target}"
  else
    missing_ocr+=("${base_size}")
  fi
done

if (( ${#missing_ocr[@]} )); then
  echo "[tables-2-3] generating missing reader-independent OCR caches"
  run_container 0 hf download deepseek-ai/DeepSeek-OCR \
    --revision "${dsocr_revision}" \
    >"${runs_dir}/logs/download-deepseek-ocr.log" 2>&1
  export VTC_MODEL_DEEPSEEK_OCR="${dsocr_path}"
  cache_pids=()
  cache_names=()
  cache_gpu=5
  for base_size in "${missing_ocr[@]}"; do
    name="ocr-cache-b${base_size}"
    run_container "${cache_gpu}" python -u run_dsocr_reconstruct.py \
      --dataset longmemeval \
      --data_path /runs/data/longmemeval_oracle.json \
      --limit 500 --shuffle --seed 0 \
      --mode simple --engine vllm \
      --base_size "${base_size}" --render_size 1024 --font_size 16 \
      --model_path "${dsocr_path}" \
      --out "/runs/results/dsocr_cache_longmemeval_b${base_size}.json" \
      >"${runs_dir}/logs/${name}.log" 2>&1 &
    cache_pids+=("$!")
    cache_names+=("${name}")
    cache_gpu=$((cache_gpu + 1))
  done
  for i in "${!cache_pids[@]}"; do
    if ! wait "${cache_pids[$i]}"; then
      echo "[tables-2-3] FAILED ${cache_names[$i]}" >&2
      exit 1
    fi
  done
fi

run_variant() {
  local label="$1"
  local model_path="$2"
  local variant_results="${runs_dir}/results/${label}"
  local variant_checkpoints="${runs_dir}/checkpoints/${label}"
  mkdir -p "${variant_results}" "${variant_checkpoints}"

  local pids=()
  local names=()

  launch() {
    local name="$1"
    local gpu="$2"
    local command="$3"
    echo "[tables-2-3] launch ${name} on GPU ${gpu}"
    run_container "${gpu}" bash -lc "${command}" \
      >"${runs_dir}/logs/${name}.log" 2>&1 &
    pids+=("$!")
    names+=("${name}")
  }

  wait_for_jobs() {
    local failed=0
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        echo "[tables-2-3] completed ${names[$i]}"
      else
        echo "[tables-2-3] FAILED ${names[$i]} (see logs/${names[$i]}.log)" >&2
        failed=1
      fi
    done
    (( failed == 0 ))
  }

  for spec in "8:0" "16:1" "32:2"; do
    local factor="${spec%%:*}"
    local gpu="${spec##*:}"
    launch "${label}-softmem-a${factor}" "${gpu}" \
      "export VTC_RESULTS_DIR=/runs/results/${label} VTC_CHECKPOINT_DIR=/runs/checkpoints/${label}; DECODER='${model_path}' ASSISTANT_FACTORS=${factor} LIMIT=500 SEED=0 bash scripts/softtoken_role_aware.sh"
  done

  local ocr_gpu=4
  for size in 1024 640 512; do
    launch "${label}-ocr-reader-b${size}" "${ocr_gpu}" \
      "python -u run_validation.py --dataset longmemeval --data_path /runs/data/longmemeval_oracle.json --conditions dsocr --dsocr_cache /runs/results/dsocr_cache_longmemeval_b${size}.json --limit 500 --shuffle --seed 0 --text_model '${model_path}' --skip_judge --out /runs/results/${label}/results_longmemeval_dsocr_b${size}.json"
    if [[ "${ocr_gpu}" == 4 ]]; then
      ocr_gpu=6
    else
      ocr_gpu=7
    fi
  done

  launch "${label}-table3-raw" 5 \
    "export VTC_RESULTS_DIR=/runs/results/${label}; DECODER='${model_path}' RUN_NAME=table3_raw_${label} LIMIT=0 bash scripts/longbench_raw.sh --prompt_format chat"

  if ! wait_for_jobs; then
    return 1
  fi

  pids=()
  names=()
  for shard in 0 1 2 3 4 5 6 7; do
    local offset=$((shard * 63))
    launch "${label}-summary-shard-${shard}" "${shard}" \
      "python -u run_validation.py --dataset longmemeval --data_path /runs/data/longmemeval_oracle.json --conditions summary --offset ${offset} --limit 63 --shuffle --seed 0 --text_model '${model_path}' --skip_judge --out /runs/results/${label}/results_longmemeval_summary_shard_${shard}.json"
  done
  if ! wait_for_jobs; then
    return 1
  fi

  python "${experiment_dir}/scripts/merge_validation_shards.py" \
    --out "${variant_results}/results_longmemeval_summary.json" \
    "${variant_results}"/results_longmemeval_summary_shard_{0,1,2,3,4,5,6,7}.json

  echo "[tables-2-3] judging Table 2 outputs for ${label}"
  run_container "0,1" python -u official_longmemeval_judge.py \
    --data /runs/data/longmemeval_oracle.json \
    --revision "${judge_revision}" \
    --job "/runs/results/${label}/results_softtoken_u1_a8.json:pred" \
    --job "/runs/results/${label}/results_softtoken_u1_a16.json:pred" \
    --job "/runs/results/${label}/results_softtoken_u1_a32.json:pred" \
    --job "/runs/results/${label}/results_longmemeval_dsocr_b1024.json:dsocr_pred" \
    --job "/runs/results/${label}/results_longmemeval_dsocr_b640.json:dsocr_pred" \
    --job "/runs/results/${label}/results_longmemeval_dsocr_b512.json:dsocr_pred" \
    --job "/runs/results/${label}/results_longmemeval_summary.json:summary_pred" \
    >"${runs_dir}/logs/${label}-official-judge.log" 2>&1
}

run_variant base "${base_path}"
run_variant instruct "${instruct_path}"

python "${experiment_dir}/scripts/summarize_tables_2_3.py" \
  --runs-dir "${runs_dir}" \
  >"${runs_dir}/logs/summarize.log" 2>&1

echo "[tables-2-3] complete: ${runs_dir}"
echo "[tables-2-3] summary: ${runs_dir}/comparison-summary.md"
