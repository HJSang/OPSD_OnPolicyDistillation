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

# Eval-only: no training epochs
total_epochs=0

# Batch sizes — train is dummy (0 epochs) but dataloader must be non-empty,
# so keep train_batch_size small enough for MATH-500 (500 examples)
train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
rollout_n=${ROLLOUT_N:-8}
tp_size=${TP_SIZE:-1}
gpu_memory_util=${GPU_MEMORY_UTIL:-0.7}

# Validation sampling
val_temperature=${VAL_TEMPERATURE:-0.7}
val_top_p=${VAL_TOP_P:-0.8}
val_top_k=${VAL_TOP_K:-20}
val_n=${VAL_N:-16}

# Instruction variant: "boxed", "dapo", or "none"
INSTRUCTION_VARIANT=${INSTRUCTION_VARIANT:-boxed}

# Reward function: "math_reward" (boxed extraction + is_equiv) or "math_dapo" (Answer: regex + exact match)
REWARD_FUNCTION=${REWARD_FUNCTION:-math_reward}

GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# ============================================================================
# Prepare data based on instruction variant
# ============================================================================

DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data"}
OUTPUT_DIR=${DATA_DIR}/eval_processed/${INSTRUCTION_VARIANT}
mkdir -p "$OUTPUT_DIR"

echo "Preparing eval data with instruction variant: ${INSTRUCTION_VARIANT}..."
python "${SRC_ROOT}/data/process_eval_data.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --instruction_variant "$INSTRUCTION_VARIANT"

VAL_AIME24=${OUTPUT_DIR}/val_aime24.parquet
VAL_AIME25=${OUTPUT_DIR}/val_aime25.parquet
VAL_MATH500=${OUTPUT_DIR}/val_math500.parquet

# verl requires train_files even for eval-only; reuse val_math500 as dummy
TRAIN_FILE=${VAL_MATH500}

# Verify data files exist
for f in "$VAL_AIME24" "$VAL_AIME25" "$VAL_MATH500"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Data file not found: $f"
        exit 1
    fi
done

echo "Data files ready:"
ls -lh "$OUTPUT_DIR"/*.parquet

# ============================================================================
# Select reward function
# ============================================================================

if [ "$REWARD_FUNCTION" = "math_reward" ]; then
    REWARD_FN_PATH="${SRC_ROOT}/rewards/math_reward.py"
elif [ "$REWARD_FUNCTION" = "math_dapo" ]; then
    REWARD_FN_PATH="${SRC_ROOT}/rewards/math_dapo.py"
else
    echo "ERROR: Unknown REWARD_FUNCTION: $REWARD_FUNCTION (expected 'math_reward' or 'math_dapo')"
    exit 1
fi

echo "Using reward function: ${REWARD_FUNCTION} from ${REWARD_FN_PATH}"

# ============================================================================
# Build experiment name and output directories
# ============================================================================

MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '_')
RUN_ID=${FLYTE_INTERNAL_EXECUTION_ID:-local}
EXP_NAME=${MODEL_NAME_SAFE}-eval-${INSTRUCTION_VARIANT}-${REWARD_FUNCTION}-${RUN_ID}

OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/outputs/eval"}
output_dir="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "$output_dir"

echo "=== Math Evaluation Configuration ==="
echo "MODEL_PATH: $MODEL_PATH"
echo "MODEL_NAME: $MODEL_NAME"
echo "INSTRUCTION_VARIANT: $INSTRUCTION_VARIANT"
echo "REWARD_FUNCTION: $REWARD_FUNCTION"
echo "total_epochs: $total_epochs (eval only)"
echo "val_temperature: $val_temperature"
echo "val_top_p: $val_top_p"
echo "val_top_k: $val_top_k"
echo "val_n: $val_n"
echo "tp_size: $tp_size"
echo "gpu_memory_util: $gpu_memory_util"
echo "EXP_NAME: $EXP_NAME"
echo "output_dir: $output_dir"
echo "======================================="

# ============================================================================
# Launch eval-only (0 epochs, val_before_train=True)
# ============================================================================

echo "Starting evaluation..."

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$TRAIN_FILE \
    data.val_files="['$VAL_MATH500','$VAL_AIME24','$VAL_AIME25']" \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    reward.custom_reward_function.path=${REWARD_FN_PATH} \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$MLFLOW_EXPERIMENT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.default_local_dir=$output_dir \
    trainer.validation_data_dir=$output_dir \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=$total_epochs

echo ""
echo "=== Evaluation completed ==="
echo "Model: $MODEL_NAME"
echo "Instruction: $INSTRUCTION_VARIANT"
echo "Reward: $REWARD_FUNCTION"
echo "Results saved to: $output_dir"
