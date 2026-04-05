# Memory Efficient OPSD On-Policy Distillation Training

Minimal training repo for math-focused distillation experiments built on top of `verl`.

Disclaimer: OPSD in this repository is developed based on our own understanding of the algorithm. The implementation is subject to change as our understanding and the codebase evolve.

This repository currently contains:

- `scripts/` for launching training, evaluation, and checkpoint conversion
- `src/opsd` for On-Policy Self-Distillation (OPSD)
- `src/data` for dataset preparation
- `src/rewards` for math reward functions
- `src/common` for shared loss and batch utilities

This trimmed copy does not include the original workflow files, tests, execution configs, or model config artifacts.

## Layout

```text
scripts/
  eval/
  grpo/
  opsd/
  utils/
src/
  common/
  data/
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

## Memory Efficiency

The training code uses several mechanisms to keep memory usage manageable on long-context math runs:

- FSDP parameter and optimizer offload. The launch scripts enable `actor.fsdp_config.param_offload=True`, `actor.fsdp_config.optimizer_offload=True`, and `ref.fsdp_config.param_offload=True` so model weights and optimizer state can be moved off GPU when inactive.
- Remove-padding execution. Training scripts set `actor_rollout_ref.model.use_remove_padding=True`, and the OPSD worker uses unpadded sequence paths so compute and memory scale with real token count instead of padded sequence length.
- Two-phase teacher/student execution for distillation. OPSD does not keep both teacher and student workloads active on GPU at the same time. The worker first runs teacher-side computation, moves cached teacher statistics or logits to CPU, offloads the teacher, and only then runs the student update step.
- Chunked divergence computation. OPSD divergence losses in `src/opsd/losses.py` process tokens in chunks instead of materializing full-vocabulary probability tensors for the whole batch at once.
- Micro-batching in the worker. OPSD splits batches using `ppo_micro_batch_size_per_gpu` and accumulates gradients across micro-batches to bound activation and logits memory.
- Dynamic batch sizing for GRPO. The main GRPO script enables `actor.use_dynamic_bsz` and caps per-GPU token counts with `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`, which is useful when response lengths vary a lot.
- Rollout memory controls. The scripts enable `rollout.free_cache_engine=True` and expose `GPU_MEMORY_UTIL` so KV-cache usage can be bounded during generation.

In practice, the biggest repo-specific savings come from the OPSD two-phase worker design, chunked loss computation, and remove-padding execution.

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
- Eval: `INSTRUCTION_VARIANT`, `REWARD_FUNCTION`, `VAL_TEMPERATURE`, `VAL_TOP_P`, `VAL_TOP_K`, `VAL_N`
