# OPSD On-Policy Distillation

Minimal training repo for math-focused distillation experiments built on top of `verl`.

This repository currently contains:

- `scripts/` for launching training, evaluation, and checkpoint conversion
- `src/opsd` for On-Policy Self-Distillation (OPSD)
- `src/eopd` for Entropy-Aware On-Policy Distillation (EOPD)
- `src/data` for dataset preparation
- `src/rewards` for math reward functions
- `src/common` for shared loss and batch utilities

This trimmed copy does not include the original workflow files, tests, execution configs, or model config artifacts.

## Layout

```text
scripts/
  eopd/
  eval/
  grpo/
  opsd/
  utils/
src/
  common/
  data/
  eopd/
  opsd/
  rewards/
```

## Environment Assumptions

The scripts assume a GPU machine with:

- Python 3
- CUDA and `nvidia-smi`
- `verl`
- `torch`
- `transformers`
- `ray`
- `hydra`
- `tensordict`

The setup scripts under `scripts/*/setup_*.sh` only do lightweight verification plus `pip install tensordict`; they do not create a full environment from scratch.

## Tested Environment

The current testing environment is:

```text
verl         0.7.0.7
torch        2.9.1.7
transformers 4.57.1
torchao      0.9.0
torchaudio   2.9.1.1
torchvision  0.24.1.10
```

## Data Layout

By default, training and eval scripts look for data under:

```text
<repo>/data
```

Expected raw inputs:

```text
data/
  DAPO-Math-17k-dedup/distinct-prompts-with-rewards.parquet
  AIME_2024/aime_2024_problems.parquet
  AIME_2025/train.jsonl
  MATH-500/test.jsonl
```

Generated files:

- `data/grpo_processed/*.parquet` from `src/data/prepare_grpo_data.py`
- `data/eval_processed/<variant>/*.parquet` from `src/data/process_eval_data.py`

## Main Entry Points

GRPO:

```bash
bash scripts/grpo/setup_grpo.sh
MODEL_PATH=/path/to/model \
MODEL_NAME=my-model \
bash scripts/grpo/train_grpo.sh
```

Native GRPO with KL:

```bash
MODEL_PATH=/path/to/model \
MODEL_NAME=my-model \
bash scripts/grpo/train_grpo_native.sh
```

Native GRPO without KL:

```bash
MODEL_PATH=/path/to/model \
MODEL_NAME=my-model \
bash scripts/grpo/train_grpo_native_no_kl.sh
```

OPSD:

```bash
bash scripts/opsd/setup_opsd.sh
MODEL_PATH=/path/to/model \
MODEL_NAME=my-model \
bash scripts/opsd/train_opsd.sh
```

EOPD:

```bash
bash scripts/eopd/setup_eopd.sh
MODEL_PATH=/path/to/model \
MODEL_NAME=my-model \
bash scripts/eopd/train_eopd.sh
```

Evaluation:

```bash
MODEL_PATH=/path/to/model \
MODEL_NAME=my-model \
INSTRUCTION_VARIANT=boxed \
REWARD_FUNCTION=math_reward \
bash scripts/eval/eval_math.sh
```

Checkpoint conversion:

```bash
CHECKPOINT_PATH=/path/to/global_step_54/actor \
bash scripts/utils/convert_checkpoint.sh
```

## Useful Environment Variables

Most training scripts accept overrides through environment variables, including:

- `MODEL_PATH`
- `MODEL_NAME`
- `DATA_DIR`
- `TRAIN_BATCH_SIZE`
- `PPO_MINI_BATCH_SIZE`
- `PPO_MICRO_BATCH_SIZE_PER_GPU`
- `LEARNING_RATE`
- `TOTAL_EPOCHS`
- `MAX_PROMPT_LENGTH`
- `MAX_RESPONSE_LENGTH`
- `ROLLOUT_N`
- `TP_SIZE`
- `GPU_MEMORY_UTIL`

Additional method-specific variables:

- OPSD: `TEACHER_MODEL_PATH`, `OPSD_LOSS_TYPE`, `OPSD_CHUNK_SIZE`, `OPSD_MAX_LENGTH`, `OPSD_REWARD_BETA`, `ENABLE_THINKING`
- EOPD: `TEACHER_MODEL_PATH`, `EOPD_CLIP_EPSILON`, `EOPD_ENTROPY_THRESHOLD`, `EOPD_TOPK`, `EOPD_N_PPO_EPOCHS`, `EOPD_CHUNK_SIZE`, `EOPD_MAX_LENGTH`
- Eval: `INSTRUCTION_VARIANT`, `REWARD_FUNCTION`, `VAL_TEMPERATURE`, `VAL_TOP_P`, `VAL_TOP_K`, `VAL_N`

## Notes

- The scripts use repo-relative path discovery; they do not rely on the old `/home/jobuser/resources/workspace` layout.
- Training outputs are currently written to `/shared/public/sharing/RLPilot/...` inside the shell scripts.
- Console logging is enabled; MLflow setup was removed from the scripts.
