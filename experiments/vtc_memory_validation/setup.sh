#!/bin/bash
# One-time environment setup for the VTC conversational-memory experiments.
#
# Stages data + models into $VTC_NFS_ROOT and builds the two Python venvs.
# Run ON A GPU POD that can reach the internal pip mirror (idev pod), e.g.:
#
#   source experiments/vtc_memory_validation/env.sh
#   bash experiments/vtc_memory_validation/setup.sh
#
# Override any path first: `export VTC_NFS_ROOT=/my/nfs` etc. (see env.sh).
# Idempotent: skips steps whose outputs already exist.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

echo "=== 1. Create staging dirs on NFS ==="
mkdir -p "$VTC_DATA_DIR" "$VTC_RESULTS_DIR"

echo "=== 2. Download benchmark data (needs internet / HF) ==="
# LoCoMo + LongMemEval oracle. prepare_data.py writes into ./data; copy to NFS.
if [ ! -f "$VTC_DATA_DIR/longmemeval_oracle.json" ]; then
    python "$HERE/prepare_data.py" --data_dir "$HERE/data"
    cp -n "$HERE/data/"*.json "$VTC_DATA_DIR/" || true
else
    echo "  longmemeval_oracle.json already staged, skipping"
fi

echo "=== 3. UltraChat training corpus (2000 convs) ==="
# The encoder trains on UltraChat (zero-shot protocol). If absent, build it.
if [ ! -f "$VTC_DATA_DIR/ultrachat_train.json" ]; then
    python "$HERE/prepare_data.py" --ultrachat --n_ultrachat 2000 \
        --data_dir "$HERE/data" || {
        echo "  NOTE: add --ultrachat support to prepare_data.py, or stage"
        echo "  ultrachat_train.json manually into $VTC_DATA_DIR/"; }
    cp -n "$HERE/data/ultrachat_train.json" "$VTC_DATA_DIR/" 2>/dev/null || true
else
    echo "  ultrachat_train.json already staged, skipping"
fi

echo "=== 4. DeepSeek-OCR model ==="
if [ ! -d "$VTC_DEEPSEEK_OCR" ]; then
    echo "  Downloading DeepSeek-OCR into $VTC_DEEPSEEK_OCR ..."
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="deepseek-ai/DeepSeek-OCR",
                  local_dir="$VTC_DEEPSEEK_OCR",
                  ignore_patterns=["*.md", "*.gitattributes"])
print("done")
PY
else
    echo "  DeepSeek-OCR already staged, skipping"
fi

echo "=== 5. vLLM venv (>=0.11.2, has DeepseekOCRForCausalLM) ==="
if [ ! -x "$VTC_VLLM_PY" ]; then
    python -m venv --system-site-packages "$(dirname "$(dirname "$VTC_VLLM_PY")")"
    "$VTC_VLLM_PY" -m pip install -q "vllm==0.11.2"
    echo "  vLLM venv built at $VTC_VLLM_PY"
else
    echo "  vLLM venv already exists, skipping"
fi

echo "=== 6. (optional) legacy DeepSeek-OCR transformers venv ==="
# Only needed for the old transformers reconstruction path (superseded by vLLM).
if [ ! -x "$VTC_DSOCR_PY" ]; then
    python -m venv --system-site-packages "$(dirname "$(dirname "$VTC_DSOCR_PY")")"
    SP="$(dirname "$(dirname "$VTC_DSOCR_PY")")/lib/python3.10/site-packages"
    "$VTC_DSOCR_PY" -m pip install -q "transformers==4.46.3"
    "$VTC_DSOCR_PY" -m pip install -q --target="$SP" addict easydict
    echo "  legacy dsocr venv built at $VTC_DSOCR_PY"
else
    echo "  legacy dsocr venv already exists, skipping"
fi

echo ""
echo "=== SETUP DONE ==="
echo "Data:   $VTC_DATA_DIR"
echo "Models: $VTC_QWEN25_7B ; $VTC_DEEPSEEK_OCR"
echo "vLLM:   $VTC_VLLM_PY"
echo ""
echo "Now submit sweeps, e.g.:"
echo "  mldev run vtc_sweep -e softtoken_full_u1_zeroshot -d \$VTC_CLUSTER --crew-id \$VTC_CREW_ID"
