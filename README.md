# Memory Efficient On-Policy Distillation Training

<p align="left">
  <a href="https://arxiv.org/abs/2604.14084"><img src="https://img.shields.io/badge/TIP-arXiv%202604.14084-b31b1b.svg" alt="TIP arXiv"></a>
  <a href="https://arxiv.org/abs/2603.11178"><img src="https://img.shields.io/badge/PACED-arXiv%202603.11178-b31b1b.svg" alt="PACED arXiv"></a>
  <a href="https://arxiv.org/abs/2605.12483"><img src="https://img.shields.io/badge/Sparse--to--Dense-arXiv%202605.12483-b31b1b.svg" alt="Sparse-to-Dense arXiv"></a>
  <a href="https://github.com/HJSang/OPSD_OnPolicyDistillation"><img src="https://img.shields.io/badge/GitHub-HJSang%2FOPSD__OnPolicyDistillation-181717.svg?logo=github" alt="GitHub"></a>
</p>

Minimal training repo for on-policy distillation (OPD) experiments built on
top of `verl`. It is the shared training base for three papers on token-,
problem-, and reward-level efficiency in OPD — see **Papers & Results**
below for what each one measures and how to cite it.

## Papers & Results

### TIP: Token Importance in On-Policy Distillation

[arXiv:2604.14084](https://arxiv.org/abs/2604.14084) · [PDF](https://arxiv.org/pdf/2604.14084)
— Yuanda Xu\*, Hejian Sang\*, Zhengze Zhou\*, Ran He\*, Zhipeng Wang, Alborz Geramifard

Not all token positions in a rollout carry equally useful learning signal.
TIP shows informative tokens live in two regions — high student entropy, and
low entropy with high teacher–student divergence (overconfident-and-wrong) —
and combines both into a Soft-OR selection score. Entropy-based retention of
50% of tokens matches or exceeds full-token training while cutting peak
training memory by up to 47%; isolating the low-entropy/high-divergence
region alone, training on under 10% of tokens nearly matches full-token
baselines.

**Main results** (accuracy %, mean@16 ± std; `Soft-OR` combines entropy and divergence):

| Model pair | Benchmark | Baseline 100% | Entropy-only 50% | Entropy-only 20% | Soft-OR 50% | Soft-OR 20% |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B → 4B | MATH-500 | 76.7 ± 0.7 | 78.6 ± 0.6 | 74.1 ± 0.9 | **79.1 ± 0.8** | 77.6 ± 0.7 |
| Qwen3-8B → 4B | AIME'24 | 21.9 ± 1.2 | 23.8 ± 1.3 | 22.5 ± 1.1 | **25.7 ± 1.4** | 24.5 ± 1.2 |
| Qwen3-8B → 4B | AIME'25 | 19.4 ± 1.1 | 20.7 ± 1.3 | 21.5 ± 1.2 | 21.9 ± 1.2 | **23.2 ± 1.2** |
| Llama-70B → 8B | MATH-500 | 71.0 ± 0.7 | 74.0 ± 0.8 | 73.6 ± 0.7 | **74.7 ± 1.0** | 74.2 ± 0.7 |
| Llama-70B → 8B | AIME'24 | 21.5 ± 1.1 | 25.3 ± 1.5 | 18.8 ± 1.3 | **26.0 ± 1.4** | 21.0 ± 1.5 |
| Llama-70B → 8B | AIME'25 | 4.9 ± 0.9 | 7.5 ± 1.1 | 10.0 ± 1.2 | **11.5 ± 1.1** | 10.9 ± 1.4 |
| Qwen2.5-14B → 1.5B | MATH-500 | 55.1 ± 0.9 | 54.9 ± 0.9 | 54.0 ± 0.9 | **56.2 ± 1.2** | 55.8 ± 0.9 |
| Qwen2.5-14B → 1.5B | AIME'24 | 2.4 ± 0.7 | 3.3 ± 1.4 | 4.6 ± 1.3 | 3.8 ± 1.2 | **5.0 ± 1.3** |
| Qwen2.5-14B → 1.5B | AIME'25 | 2.1 ± 0.9 | 1.0 ± 0.5 | 1.0 ± 0.6 | 1.5 ± 0.7 | **1.8 ± 0.6** |

Generalizes beyond math: on **DeepPlanning** agentic planning (Qwen3-1.7B
student, Avg@16), training on only 20% of overconfident (Q3) tokens surpasses
full-token OPD for both a 14B teacher (12.6% vs. 11.7%) and a 32B teacher
(13.6% vs. 12.8%).

```bibtex
@article{xu2026tip,
  title   = {TIP: Token Importance in On-Policy Distillation},
  author  = {Xu, Yuanda and Sang, Hejian and Zhou, Zhengze and He, Ran and Wang, Zhipeng and Geramifard, Alborz},
  journal = {arXiv preprint arXiv:2604.14084},
  year    = {2026}
}
```

### PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence

[arXiv:2603.11178](https://arxiv.org/abs/2603.11178) · [PDF](https://arxiv.org/pdf/2603.11178)
— Yuanda Xu\*, Hejian Sang\*, Zhengze Zhou\*, Ran He\*, Zhipeng Wang

Standard distillation treats every training problem equally, wasting compute
on problems the student has already mastered or still cannot solve. PACED
weights each problem by `w(p) = p(1-p)` (student empirical pass rate `p`),
concentrating gradient signal-to-noise at the frontier of student competence.

**Distillation track** (Qwen3-8B-GRPO → Qwen3-1.7B, forward-KL family, 8-sample mean accuracy %):

| Method | MATH-500 | AIME 2024 | AIME 2025 |
|---|---:|---:|---:|
| Base | 69.4 ± 0.4 | 11.5 ± 0.9 | 7.6 ± 0.7 |
| Forward KL (unweighted) | 76.8 ± 0.3 | 21.2 ± 1.3 | 17.0 ± 0.9 |
| Hard Filter Forward KL | 78.5 ± 0.6 | 23.7 ± 0.9 | 18.8 ± 0.6 |
| AKL | 77.6 ± 0.4 | 23.9 ± 1.2 | 19.1 ± 0.8 |
| **PACED Forward KL** | **79.4 ± 0.5** | **25.1 ± 1.0** | **20.6 ± 0.7** |

**Self-distillation track** (Qwen2.5-Math-7B-Instruct, reverse-KL family, 8-sample mean accuracy %):

| Method | MATH-500 | AIME 2024 | AIME 2025 |
|---|---:|---:|---:|
| Base | 83.9 ± 0.6 | 19.6 ± 1.0 | 11.5 ± 0.7 |
| Reverse KL (unweighted) | 88.9 ± 0.5 | 25.3 ± 1.2 | 16.9 ± 1.1 |
| Hard Filter Reverse KL | 92.0 ± 0.5 | 28.9 ± 1.3 | 22.0 ± 0.9 |
| AKL | 91.4 ± 0.5 | 28.2 ± 0.8 | 21.5 ± 0.6 |
| **PACED Reverse KL** | **93.7 ± 0.6** | **31.6 ± 1.1** | **25.1 ± 0.7** |

A two-stage forward-then-reverse KL schedule performs best overall, and PACED
retains MMLU/forgetting on par with the best hard-filtering baseline (Tables
4–5 of the paper) while improving reasoning accuracy.

```bibtex
@article{xu2026paced,
  title   = {PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence},
  author  = {Xu, Yuanda and Sang, Hejian and Zhou, Zhengze and He, Ran and Wang, Zhipeng},
  journal = {arXiv preprint arXiv:2603.11178},
  year    = {2026}
}
```

### Beyond GRPO and On-Policy Distillation: An Empirical Sparse-to-Dense Reward Principle for LLM Post-Training

[arXiv:2605.12483](https://arxiv.org/abs/2605.12483) · [PDF](https://arxiv.org/pdf/2605.12483)
— Hejian Sang\*, Yuanda Xu\*, Zhengze Zhou\*, Ran He\*, Zhipeng Wang, Alborz Geramifard

When labeled verifiable data is the binding constraint, scarce sparse-reward
RL is most useful upstream on a strong teacher; dense token-level distillation
is what compresses that behavior back into a small deployment model. The
paper formalizes this as a three-stage workflow — teacher RL, forward-KL
warmup, on-policy distillation (OPD), then optional post-bridge student RL —
and shows every stage is load-bearing.

**Direct-GRPO baseline** across Qwen3 scales (avg@16, %) — the 1.7B row is the
deployment-student endpoint the workflow must beat:

| Model | MATH | AIME 2024 | AIME 2025 |
|---|---:|---:|---:|
| Qwen3-1.7B (cold GRPO) | 75.9 ± 0.9 | 19.8 ± 1.4 | 17.1 ± 0.9 |
| Qwen3-8B | 88.4 ± 0.8 | 47.7 ± 1.5 | 36.7 ± 1.2 |
| Qwen3-14B | 89.5 ± 0.7 | 47.1 ± 1.2 | 39.0 ± 0.9 |

**Full workflow vs. cold GRPO**, Qwen3-1.7B deployment student (MATH, avg@16, %):

| Teacher | Full workflow (bridge + Stage 3) | Cold GRPO (no bridge) | Δ |
|---|---:|---:|---:|
| RL'd Qwen3-8B | 78.5 ± 0.9 | 75.9 ± 0.9 | +2.6 |
| RL'd Qwen3-14B | 78.7 ± 1.1 | 75.9 ± 0.9 | +2.8 |

Every stage matters: removing teacher-side RL collapses the student to
71.5–72.8% MATH (below cold GRPO); removing the forward-KL warmup costs
1.5–1.7 points; and Stage-3 student RL after the bridge adds +2.4 to +3.1
points that a replay control (same number of updates, no new labels) cannot
reproduce (≤0.3 points). The teacher-RL + distillation setup outperforms
directly training small models with GRPO/RL.

```bibtex
@article{sang2026sparsetodense,
  title   = {Beyond GRPO and On-Policy Distillation: An Empirical Sparse-to-Dense Reward Principle for Language-Model Post-Training},
  author  = {Sang, Hejian and Xu, Yuanda and Zhou, Zhengze and He, Ran and Wang, Zhipeng and Geramifard, Alborz},
  journal = {arXiv preprint arXiv:2605.12483},
  year    = {2026}
}
```

## Related Repositories

- [`HJSang/LatentPress`](https://github.com/HJSang/LatentPress) —
  **LatentPress: Context Compression Beyond Text and Vision**
  ([arXiv:2609.01507](https://arxiv.org/abs/2609.01507)), a separate research
  line by one of this repo's authors. LatentPress writes long context into
  continuous soft tokens that a frozen decoder reads directly, rather than
  distilling behavior between a teacher and a student — an orthogonal
  compression axis (representation of context) to OPD's compression axis
  (which tokens carry learning signal during distillation).

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

## Acknowledgements

Training builds on [`verl`](https://github.com/volcengine/verl) /
HybridFlow (Sheng et al., 2025) for distributed RL and rollout
infrastructure.
