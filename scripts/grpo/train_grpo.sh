#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

ulimit -n 65535
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-39999 -n 1)}

echo "PYTHONPATH: $PYTHONPATH"

# MLflow setup removed; use console logging only.

# ============================================================================
# Configuration (from environment variables, with defaults)
# ============================================================================

MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_PATH")}

# GRPO hyperparameters
train_batch_size=${TRAIN_BATCH_SIZE:-512}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
learning_rate=${LEARNING_RATE:-1e-6}
total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
rollout_n=${ROLLOUT_N:-8}
tp_size=${TP_SIZE:-1}
gpu_memory_util=${GPU_MEMORY_UTIL:-0.7}
kl_loss_coef=${KL_LOSS_COEF:-0.0}
entropy_coeff=${ENTROPY_COEFF:-0}

# DAPO-specific: asymmetric clipping (clip_ratio_low < clip_ratio_high)
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
clip_ratio_c=${CLIP_RATIO_C:-10.0}

# DAPO-specific: dynamic batch size (variable-length sequence packing)
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2 ))
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 3 ))

# DAPO-specific: token-level loss aggregation
loss_agg_mode=${LOSS_AGG_MODE:-token-mean}

# DAPO-specific: no KL penalty (rely on clip instead)
use_kl_loss=False
use_kl_in_reward=False

# Sampling: high temperature for exploration during training, lower for val
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_temperature=${VAL_TEMPERATURE:-0.7}
val_top_p=${VAL_TOP_P:-0.8}
val_top_k=${VAL_TOP_K:-20}

# Data split ratio
dapo_train_ratio=${DAPO_TRAIN_RATIO:-0.8}

GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# ============================================================================
# Prepare data: split DAPO 80/20 and process validation sets
# ============================================================================

DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data"}
OUTPUT_DIR=${DATA_DIR}/grpo_processed

echo "Preparing GRPO data (DAPO ${dapo_train_ratio} train split)..."
python "${SRC_ROOT}/data/prepare_grpo_data.py" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --train-ratio "$dapo_train_ratio"

TRAIN_FILE=${OUTPUT_DIR}/train.parquet
VAL_DAPO=${OUTPUT_DIR}/val_dapo.parquet
VAL_AIME24=${OUTPUT_DIR}/val_aime24.parquet
VAL_AIME25=${OUTPUT_DIR}/val_aime25.parquet
VAL_MATH500=${OUTPUT_DIR}/val_math500.parquet

# Verify data files exist
for f in "$TRAIN_FILE" "$VAL_DAPO" "$VAL_AIME24" "$VAL_AIME25" "$VAL_MATH500"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Data file not found: $f"
        exit 1
    fi
done

echo "Data files ready:"
ls -lh "$OUTPUT_DIR"/*.parquet

# ============================================================================
# Build experiment name and output directories
# ============================================================================

MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '_')
RUN_ID=${RUN_ID:-local}
EXP_NAME=${MODEL_NAME_SAFE}-${RUN_ID}-GRPO-lr${learning_rate}-bs${train_batch_size}-n${rollout_n}

OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/outputs"}
output_dir="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "$output_dir"

echo "=== GRPO Training Configuration ==="
echo "MODEL_PATH: $MODEL_PATH"
echo "MODEL_NAME: $MODEL_NAME"
echo "train_batch_size: $train_batch_size"
echo "ppo_mini_batch_size: $ppo_mini_batch_size"
echo "learning_rate: $learning_rate"
echo "total_epochs: $total_epochs"
echo "max_prompt_length: $max_prompt_length"
echo "max_response_length: $max_response_length"
echo "rollout_n: $rollout_n"
echo "tp_size: $tp_size"
echo "gpu_memory_util: $gpu_memory_util"
echo "--- DAPO features ---"
echo "clip_ratio_low: $clip_ratio_low"
echo "clip_ratio_high: $clip_ratio_high"
echo "use_dynamic_bsz: $use_dynamic_bsz"
echo "loss_agg_mode: $loss_agg_mode"
echo "temperature: $temperature"
echo "val_top_p: $val_top_p"
echo "EXP_NAME: $EXP_NAME"
echo "output_dir: $output_dir"
echo "====================================="

# ============================================================================
# Launch GRPO training
# ============================================================================

echo "Starting GRPO training..."

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    data.train_files=$TRAIN_FILE \
    data.val_files="['$VAL_DAPO','$VAL_MATH500','$VAL_AIME24','$VAL_AIME25']" \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$learning_rate \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=grpo \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.default_local_dir=$output_dir \
    trainer.validation_data_dir=$output_dir \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs

echo ""
echo "=== GRPO training completed ==="
echo "Model: $MODEL_NAME"
echo "Checkpoints and results saved to: $output_dir"
