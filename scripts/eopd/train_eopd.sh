#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_ROOT="${REPO_ROOT}/src"

ulimit -n 65535

export PYTHONPATH="${SRC_ROOT}:$PYTHONPATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "PYTHONPATH: $PYTHONPATH"

# MLflow setup removed; use console logging only.

# ============================================================================
# Configuration (from environment variables, with defaults)
# ============================================================================

MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_PATH")}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-}

# Training hyperparameters (paper defaults: batch=128, mini=32, lr=3e-6)
train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
learning_rate=${LEARNING_RATE:-3e-6}
total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
rollout_n=${ROLLOUT_N:-1}
tp_size=${TP_SIZE:-1}
gpu_memory_util=${GPU_MEMORY_UTIL:-0.7}

# EOPD-specific hyperparameters
eopd_clip_epsilon=${EOPD_CLIP_EPSILON:-0.2}
eopd_entropy_threshold=${EOPD_ENTROPY_THRESHOLD:-0.8}
eopd_topk=${EOPD_TOPK:-16}
eopd_n_ppo_epochs=${EOPD_N_PPO_EPOCHS:-4}
eopd_chunk_size=${EOPD_CHUNK_SIZE:-512}
eopd_max_length=${EOPD_MAX_LENGTH:-16384}

# Sampling
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}

# Paper uses Qwen3-8B teacher WITHOUT thinking mode
ENABLE_THINKING=${ENABLE_THINKING:-False}
# Paper evaluation: temperature=1.0, top_p=0.8
val_temperature=${VAL_TEMPERATURE:-1.0}
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
RUN_ID=${FLYTE_INTERNAL_EXECUTION_ID:-local}
EXP_NAME=${MODEL_NAME_SAFE}-${RUN_ID}-EOPD-tau${eopd_entropy_threshold}-k${eopd_topk}-eps${eopd_clip_epsilon}-lr${learning_rate}-bs${train_batch_size}-n${rollout_n}

OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/outputs"}
output_dir="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "$output_dir"

echo "=== EOPD Training Configuration ==="
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
echo "--- EOPD features ---"
echo "eopd_clip_epsilon: $eopd_clip_epsilon"
echo "eopd_entropy_threshold: $eopd_entropy_threshold"
echo "eopd_topk: $eopd_topk"
echo "eopd_n_ppo_epochs: $eopd_n_ppo_epochs"
echo "eopd_chunk_size: $eopd_chunk_size"
echo "eopd_max_length: $eopd_max_length"
echo "temperature: $temperature"
echo "val_temperature: $val_temperature"
echo "EXP_NAME: $EXP_NAME"
echo "output_dir: $output_dir"
echo "====================================="

# ============================================================================
# Launch EOPD training via Hydra
# ============================================================================

echo "Starting EOPD training (tau=${eopd_entropy_threshold}, k=${eopd_topk})..."

python -m eopd.main_eopd \
    --config-path "${SRC_ROOT}/eopd/config" \
    --config-name eopd_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files="['$VAL_DAPO','$VAL_MATH500','$VAL_AIME24','$VAL_AIME25']" \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
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
    ${TEACHER_MODEL_PATH:+"+actor_rollout_ref.ref.model.path=$TEACHER_MODEL_PATH"} \
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
    eopd.clip_epsilon=${eopd_clip_epsilon} \
    eopd.entropy_threshold=${eopd_entropy_threshold} \
    eopd.topk=${eopd_topk} \
    eopd.n_ppo_epochs=${eopd_n_ppo_epochs} \
    eopd.chunk_size=${eopd_chunk_size} \
    eopd.max_length=${eopd_max_length} \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$MLFLOW_EXPERIMENT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.default_local_dir=$output_dir \
    +trainer.validation_data_dir=$output_dir \
    trainer.val_before_train=False \
    trainer.log_val_generations=10 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs

echo ""
echo "=== EOPD training completed ==="
echo "Model: $MODEL_NAME"
echo "Entropy threshold: $eopd_entropy_threshold"
echo "Top-k: $eopd_topk"
echo "Checkpoints and results saved to: $output_dir"
