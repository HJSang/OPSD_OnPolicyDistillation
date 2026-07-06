#!/bin/bash
# Central configuration for the VTC conversational-memory experiments.
# Source this before running anything, or override any var in your shell.
#
#   source experiments/vtc_memory_validation/env.sh
#
# In a NEW environment, override these (e.g. export VTC_NFS_ROOT=/my/path)
# then run setup.sh to stage data/models/venvs.

# --- storage roots ---------------------------------------------------------
# Shared/NFS staging area for data, models, venvs, and job results.
export VTC_NFS_ROOT="${VTC_NFS_ROOT:-/shared/public/sharing/vtc_memory}"

# Where base models live (used to resolve the model registry).
export VTC_MODELS_ROOT="${VTC_MODELS_ROOT:-/shared/public/models}"
export VTC_ELR_MODELS_ROOT="${VTC_ELR_MODELS_ROOT:-/shared/public/elr-models}"

# --- model paths (override if your models live elsewhere) ------------------
export VTC_QWEN25_7B="${VTC_QWEN25_7B:-$VTC_MODELS_ROOT/Qwen/Qwen2.5-7B-Instruct}"
export VTC_DEEPSEEK_OCR="${VTC_DEEPSEEK_OCR:-$VTC_NFS_ROOT/DeepSeek-OCR}"

# --- python venvs ----------------------------------------------------------
# vLLM venv (>=0.11.2, has DeepseekOCRForCausalLM) for fast OCR reconstruction.
export VTC_VLLM_PY="${VTC_VLLM_PY:-$HOME/vllm_env/bin/python}"
# Optional legacy DeepSeek-OCR venv (transformers 4.46) — only for the old
# transformers reconstruction path (superseded by vLLM).
export VTC_DSOCR_PY="${VTC_DSOCR_PY:-$HOME/dsocr_env/bin/python}"

# --- cluster / submission (mldev) ------------------------------------------
export VTC_CLUSTER="${VTC_CLUSTER:-prod-lva1-k8s-2}"
export VTC_CREW_ID="${VTC_CREW_ID:-3330}"

# --- derived paths ---------------------------------------------------------
export VTC_DATA_DIR="${VTC_DATA_DIR:-$VTC_NFS_ROOT/data}"
export VTC_RESULTS_DIR="${VTC_RESULTS_DIR:-$VTC_NFS_ROOT/results}"

echo "[env] VTC_NFS_ROOT=$VTC_NFS_ROOT"
echo "[env] VTC_QWEN25_7B=$VTC_QWEN25_7B"
echo "[env] VTC_DEEPSEEK_OCR=$VTC_DEEPSEEK_OCR"
echo "[env] VTC_VLLM_PY=$VTC_VLLM_PY"
echo "[env] VTC_CLUSTER=$VTC_CLUSTER"
