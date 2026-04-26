# Memory Efficient On-Policy Distillation Training

Minimal training repo for on-policy distillation experiments built on top of `verl`.

## Papers

This repository is related to the following papers:

- [TIP: Token Importance in On-Policy Distillation](https://arxiv.org/abs/2604.14084) ([PDF](https://arxiv.org/pdf/2604.14084))
  - Studies which token positions carry the most useful learning signal in OPD.
  - Introduces the TIP view of token importance based on student entropy and teacher-student divergence.

- [PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence](https://arxiv.org/abs/2603.11178) ([PDF](https://arxiv.org/pdf/2603.11178))
  - Studies sample importance for distillation and self-distillation at the problem level.
  - Proposes weighting problems by student empirical pass rate, emphasizing the frontier of student competence.
  - A two-stage forward-then-reverse KL schedule leads to the best performance.

> **TODO:** Add OPSD (On-Policy Self-Distillation) support -- same model as teacher and student, where the teacher sees ground truth context. Currently only OPD (separate teacher model) is included.

## OPD: On-Policy Distillation with Separate Teacher

A separate (typically bigger) teacher model and a trainable student model see the same input sequences. The teacher produces better distributions naturally; no ground-truth injection is needed.

- Entry point: `python -m opd.main_opd`
- Requires `TEACHER_MODEL_PATH` environment variable
- Batch construction: `build_opd_batch` (trainer entry point) prefers pre-tokenized `batch["prompts"]` + `response_mask` so training matches rollout inputs; falls back to `raw_prompt` + chat template only if prompts are absent
- `build_opd_batch_multiturn` / `build_opd_batch_from_verl_batch` remain as thin aliases for the prompts-only and raw-prompt-only paths
- Supports reward-weighted distillation via `opd.reward_beta` config

## Multi-turn Agent-loop Support

OPD supports multi-turn agent-loop rollouts where the response contains interleaved LLM-generated tokens and tool/environment tokens:

- The trainer preserves the agent-loop `response_mask` (1=LLM, 0=tool) instead of recomputing it
- The batch builder uses `response_mask` as the per-token loss mask so distillation only targets LLM-generated spans
- `build_opd_batch` uses pre-tokenized prompt IDs from `batch["prompts"]` when present for exact prompt matching

Multi-turn diagnostics are logged: `tool_mask/llm_tokens`, `tool_mask/tool_tokens`, `tool_mask/tool_ratio`, `num_turns/*`.

## Layout

```text
scripts/
  eval/
  grpo/
  opd/          # OPD training scripts (separate teacher)
  utils/
src/
  common/       # Shared batch builder
  data/
  opd/          # OPD module (separate teacher model)
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
- Remove-padding execution. Training scripts set `actor_rollout_ref.model.use_remove_padding=True`, and the OPD worker uses unpadded sequence paths so compute and memory scale with real token count instead of padded sequence length.
- Two-phase teacher/student execution for distillation. OPD does not keep both teacher and student workloads active on GPU at the same time. The worker first runs teacher-side computation, moves cached teacher statistics or logits to CPU, offloads the teacher, and only then runs the student update step.
- Chunked divergence computation. OPD divergence losses in `src/opd/losses.py` process tokens in chunks instead of materializing full-vocabulary probability tensors for the whole batch at once.
- Micro-batching in the worker. OPD splits batches using `ppo_micro_batch_size_per_gpu` and accumulates gradients across micro-batches to bound activation and logits memory.
- Dynamic batch sizing for GRPO. The main GRPO script enables `actor.use_dynamic_bsz` and caps per-GPU token counts with `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`, which is useful when response lengths vary a lot.
- Rollout memory controls. The scripts enable `rollout.free_cache_engine=True` and expose `GPU_MEMORY_UTIL` so KV-cache usage can be bounded during generation.

In practice, the biggest repo-specific savings come from the OPD two-phase worker design, chunked loss computation, and remove-padding execution.

## Distillation Implementation

OPD (`src/opd/opd_worker.py`) uses a two-phase update:

1. **Phase 1 (Teacher):** Load the teacher (`ref`) model, run teacher forwards for all micro-batches, cache teacher logits on CPU, offload teacher.
2. **Phase 2 (Student):** Load the student (`actor`) model and optimizer, run student forward + divergence loss + backward using cached teacher logits.

This avoids keeping both teacher and student compute active on GPU at the same time during the update step.

OPD supports three divergence types (`reverse_kl`, `forward_kl`, `jsd`), chunk-wise loss computation, and per-sample reward weighting.

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

OPD (separate teacher, single-turn math):

```bash
bash scripts/opd/setup_opd.sh
MODEL_PATH=/path/to/student_model \
TEACHER_MODEL_PATH=/path/to/teacher_model \
MODEL_NAME=my-model \
bash scripts/opd/train_opd.sh
```

OPD (separate teacher, multi-turn agent with tool calls):

```bash
bash scripts/opd/setup_opd.sh
MODEL_PATH=/path/to/student_model \
TEACHER_MODEL_PATH=/path/to/teacher_model \
DATABASE_DIR=/path/to/tool/database \
MODEL_NAME=my-model \
bash scripts/opd/train_opd_agent.sh
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

OPD-specific variables:

- `TEACHER_MODEL_PATH` (required)
- `OPD_LOSS_TYPE`
- `OPD_CHUNK_SIZE`
- `OPD_MAX_LENGTH`
- `OPD_REWARD_BETA`
- `ENABLE_THINKING`

OPD agent additional variables:

- `ENABLE_TOOLS`
- `MAX_ASSISTANT_TURNS`
- `MAX_TOOL_RESPONSE_LENGTH`
- `TOOL_FORMAT`
- `AGENT_NUM_WORKERS`
- `DATABASE_DIR`

Eval variables:

- `INSTRUCTION_VARIANT`
- `REWARD_FUNCTION`
- `VAL_TEMPERATURE`
- `VAL_TOP_P`
- `VAL_TOP_K`
- `VAL_N`
