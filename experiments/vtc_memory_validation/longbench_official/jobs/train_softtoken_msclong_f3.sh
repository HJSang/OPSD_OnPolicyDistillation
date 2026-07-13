#!/usr/bin/env bash
set -euo pipefail

cd /home/jobuser/resources/experiments/vtc_memory_validation

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python softtoken/train.py \
  --decoder qwen3-8b \
  --dataset ultrachat \
  --data /shared/public/sharing/vtc_memory/data/msc_long_train.json \
  --factor 3 \
  --mode simple \
  --pool mean \
  --enc_layers 2 \
  --max_len 2048 \
  --n_chunks 400 \
  --long_context \
  --max_mem_tokens 8192 \
  --min_mem_tokens 512 \
  --target_len 256 \
  --enc_window 512 \
  --batch_size 1 \
  --steps 1000 \
  --lr 1e-4 \
  --fkl_weight 1.0 \
  --save softtoken/ckpt_msclong_distill_qwen3_8b_f3.pt
