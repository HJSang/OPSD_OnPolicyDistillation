#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

ulimit -n 65535

export PYTHONPATH="${SRC_ROOT}:$PYTHONPATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-39999 -n 1)}

echo "PYTHONPATH: $PYTHONPATH"

# ============================================================================
# Configuration (from environment variables, with defaults)
# ============================================================================

MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_PATH")}
# Teacher model (typically bigger/stronger). Both teacher and student see the same input.
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:?TEACHER_MODEL_PATH environment variable is required}

# Training hyperparameters
train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
learning_rate=${LEARNING_RATE:-1e-6}
total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
rollout_n=${ROLLOUT_N:-1}
tp_size=${TP_SIZE:-1}
gpu_memory_util=${GPU_MEMORY_UTIL:-0.7}

# OPD-specific: divergence type and chunk size
opd_loss_type=${OPD_LOSS_TYPE:-reverse_kl}
opd_chunk_size=${OPD_CHUNK_SIZE:-256}
opd_max_length=${OPD_MAX_LENGTH:-16384}
# Reward-weighted distillation: set to a positive float to enable (e.g., 0.1, 0.5, 1.0)
opd_reward_beta=${OPD_REWARD_BETA:-}

# Sampling: high temperature for exploration during student rollout
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}

# Qwen3 recommended params for validation
# Default: 0.6 with thinking enabled, 0.7 without
ENABLE_THINKING=${ENABLE_THINKING:-True}
if [ "$ENABLE_THINKING" = "True" ]; then
    val_temperature=${VAL_TEMPERATURE:-0.6}
else
    val_temperature=${VAL_TEMPERATURE:-0.7}
fi
val_top_p=${VAL_TOP_P:-0.8}
val_top_k=${VAL_TOP_K:-20}

# Data split ratio
dapo_train_ratio=${DAPO_TRAIN_RATIO:-0.8}
val_before_train=${VAL_BEFORE_TRAIN:-False}

GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# ============================================================================
# Prepare data: split DAPO 80/20 and process validation sets
# ============================================================================

DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data"}
OUTPUT_DIR=${DATA_DIR}/grpo_processed

echo "Preparing data (DAPO ${dapo_train_ratio} train split)..."
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
if [ "$ENABLE_THINKING" = "True" ]; then
    THINK_TAG="thinking"
else
    THINK_TAG="nothink"
fi
REWARD_TAG=""
if [ -n "$opd_reward_beta" ]; then
    REWARD_TAG="-rwbeta${opd_reward_beta}"
fi
TEACHER_NAME=$(basename "$(dirname "$TEACHER_MODEL_PATH")")
TEACHER_TAG="-teacher-${TEACHER_NAME}"
EXP_NAME=${MODEL_NAME_SAFE}-${RUN_ID}-OPD-${opd_loss_type}${REWARD_TAG}${TEACHER_TAG}-${THINK_TAG}-lr${learning_rate}-bs${train_batch_size}-n${rollout_n}

OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/outputs"}
output_dir="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "$output_dir"

# Background GPU memory monitor — survives Python OOM kills
# Logs every 2s so we can see memory at the exact moment of OOM
GPU_MONITOR_LOG="${output_dir}/gpu_memory_monitor.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,memory.total,utilization.gpu --format=csv -l 2 > "${GPU_MONITOR_LOG}" 2>&1 &
GPU_MONITOR_PID=$!
echo "GPU monitor started (PID: $GPU_MONITOR_PID), logging to ${GPU_MONITOR_LOG}"
trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT

echo "=== OPD Training Configuration ==="
echo "MODEL_PATH: $MODEL_PATH"
echo "MODEL_NAME: $MODEL_NAME"
echo "train_batch_size: $train_batch_size"
echo "ppo_mini_batch_size: $ppo_mini_batch_size"
echo "learning_rate: $learning_rate"
echo "total_epochs: $total_epochs"
echo "max_prompt_length: $max_prompt_length"
echo "max_response_length: $max_response_length"
echo "tp_size: $tp_size"
echo "gpu_memory_util: $gpu_memory_util"
echo "--- OPD features ---"
echo "opd_loss_type: $opd_loss_type"
echo "opd_chunk_size: $opd_chunk_size"
echo "opd_max_length: $opd_max_length"
echo "opd_reward_beta: ${opd_reward_beta:-disabled}"
echo "temperature: $temperature"
echo "val_temperature: $val_temperature"
echo "val_top_p: $val_top_p"
echo "val_top_k: $val_top_k"
echo "EXP_NAME: $EXP_NAME"
echo "output_dir: $output_dir"
echo "====================================="

# ============================================================================
# Launch OPD training via Hydra
# ============================================================================

echo "Starting OPD training (loss_type=${opd_loss_type})..."

python -m opd.main_opd \
    --config-path "${SRC_ROOT}/opd/config" \
    --config-name opd_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files="['$VAL_MATH500','$VAL_AIME24','$VAL_AIME25']" \
    data.return_raw_chat=True \
    $(if [[ "$MODEL_NAME" == *"Qwen"* || "$MODEL_NAME" == *"qwen"* ]]; then echo "+data.apply_chat_template_kwargs.enable_thinking=False"; fi) \
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
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.model.path=$TEACHER_MODEL_PATH \
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
    opd.loss_type=${opd_loss_type} \
    opd.chunk_size=${opd_chunk_size} \
    opd.max_length=${opd_max_length} \
    ${opd_reward_beta:+"opd.reward_beta=$opd_reward_beta"} \
    reward.custom_reward_function.path="${SRC_ROOT}/rewards/math_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.default_local_dir=$output_dir \
    +trainer.validation_data_dir=$output_dir \
    trainer.val_before_train=${val_before_train} \
    trainer.log_val_generations=10 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs

echo ""
echo "=== OPD training completed ==="
echo "Model: $MODEL_NAME"
echo "Loss type: $opd_loss_type"
echo "Checkpoints and results saved to: $output_dir"
