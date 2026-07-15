# Conversational-Memory Compression

This experiment evaluates several ways to fit long conversation histories into
an LLM context:

| Method | Context representation |
|---|---|
| `raw` | Original text |
| `summary` | LLM-generated factual summary |
| `vtc` | Text rendered as images for a vision-language model |
| `dsocr` | Text rendered as images and reconstructed by DeepSeek-OCR |
| `softtoken simple` | Uniform learned pooling |
| `softtoken role-aware` | User tokens preserved; assistant tokens pooled |

## Requirements

Use a CUDA machine with enough memory for the selected decoder.

```bash
cd experiments/vtc_memory_validation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Models are downloaded by `transformers` from Hugging Face unless an alias is
overridden with an environment variable.

### Docker GPU environment

The reproducible GPU image is based on the SLIME/VERL training stack used by
the original jobs:

| Component | Version |
|---|---|
| Base image | `slimerl/slime@sha256:9b548a94...` |
| Python | 3.12.3 |
| CUDA toolkit/runtime | 12.9.1 |
| cuDNN (PyTorch runtime) | 9.16.0 |
| NCCL (PyTorch runtime) | 2.27.5 |
| PyTorch | 2.9.1+cu129 |
| SGLang | 0.5.9 |
| Ray | 2.54.0 |
| vLLM (OCR/judge) | 0.11.2 |
| Transformers | 4.57.1 |

This stack was validated on NVIDIA B200 GPUs with driver `580.167.08`.

Build and verify it from the repository root:

```bash
experiments/vtc_memory_validation/docker/build.sh
experiments/vtc_memory_validation/docker/run.sh \
  python docker/verify_environment.py --require-gpu
```

To reproduce all rows in the paper's main table on eight GPUs, retain each
stage's log, and score every prediction with one judge-model load:

```bash
HF_TOKEN=... experiments/vtc_memory_validation/scripts/reproduce_main_table.sh
```

Artifacts are written under
`experiments/vtc_memory_validation/.docker-runs/main-table` by default.
When running from an exported source archive rather than a Git checkout, set
`VTC_GIT_COMMIT` to preserve the source revision in the run manifest.

The runner mounts the repository at `/workspace`, Hugging Face cache at
`/cache/huggingface`, and generated data, checkpoints, and results at
`experiments/vtc_memory_validation/.docker-runs`.

The default Qwen model was validated at revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Cache that exact revision with:

```bash
experiments/vtc_memory_validation/docker/run.sh \
  hf download Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28

export VTC_MODEL_QWEN2_5_7B=/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
experiments/vtc_memory_validation/docker/run.sh \
  bash scripts/raw_summary.sh
```

The default batched DeepSeek-OCR path uses vLLM in the main environment. Its
slower native fallback uses `/opt/vtc-deepseek-ocr/bin/python`, an isolated
environment with Transformers 4.46.3 and the base image's CUDA-enabled
PyTorch. The tested model revision is
`9f30c71f441d010e5429c532364a86705536c53a`:

```bash
experiments/vtc_memory_validation/docker/run.sh \
  hf download deepseek-ai/DeepSeek-OCR \
    --revision 9f30c71f441d010e5429c532364a86705536c53a

export VTC_MODEL_DEEPSEEK_OCR=/cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/9f30c71f441d010e5429c532364a86705536c53a
experiments/vtc_memory_validation/docker/run.sh \
  bash scripts/deepseek_ocr.sh
```

## Data

The scripts call `prepare_data.py` when required files are missing. To prepare
everything explicitly:

```bash
python prepare_data.py
```

This creates `data/` beside this README. The tracked 2,000-conversation
UltraChat subset is copied locally; LoCoMo and LongMemEval are downloaded from
their public sources.

## Run

Scripts can be launched from any directory:

```bash
# Raw text and summary baselines
experiments/vtc_memory_validation/scripts/raw_summary.sh

# Uniform factors 4, 8, and 16
experiments/vtc_memory_validation/scripts/softtoken_simple.sh

# User factor 1; assistant factors 8, 16, and 32
experiments/vtc_memory_validation/scripts/softtoken_role_aware.sh

# LongBench task-adapted compressor
experiments/vtc_memory_validation/scripts/longbench.sh

# LongBench raw-context baseline
experiments/vtc_memory_validation/scripts/longbench_raw.sh
```

Use environment variables to change a run without editing repository files:

```bash
DECODER=Qwen/Qwen2.5-7B-Instruct \
FACTORS="4 8" \
STEPS=200 \
LIMIT=50 \
VTC_DATA_DIR=/mnt/data/vtc \
VTC_RESULTS_DIR=/mnt/results/vtc \
VTC_CHECKPOINT_DIR=/mnt/checkpoints/vtc \
experiments/vtc_memory_validation/scripts/softtoken_simple.sh
```

The default main-table recipe uses 1,000 steps over 400 UltraChat chunks,
`MAX_LEN=2048`, `BATCH_SIZE=1`, two frozen borrowed encoder layers, mean
pooling, forward-KL weight 1.0, learning rate `1e-4`, and seed 0. Training
enables deterministic PyTorch/CUDA algorithms by default; pass
`--non_deterministic` directly to `softtoken/train.py` only when throughput is
more important than producing identical checkpoints.

The eight-GPU main-table launcher sets DeepSeek-OCR `--max_num_seqs 64`, which
was validated on a 192 GB B200. Lower this value when reproducing the OCR rows
on GPUs with less memory. OCR input size defaults to `base_size`, matching
DeepSeek-OCR's Tiny, Small, and Base processor modes.
The vLLM runner disables prefix caching per the model's official recipe and
synchronizes vLLM's import-time `crop_mode` processor default with model-side
token accounting. Without the latter, no-crop Base mode can silently generate
corrupted reconstructions.

Common path variables:

| Variable | Default |
|---|---|
| `VTC_DATA_DIR` | `experiments/vtc_memory_validation/data` |
| `VTC_RESULTS_DIR` | `experiments/vtc_memory_validation/results` |
| `VTC_CHECKPOINT_DIR` | `experiments/vtc_memory_validation/checkpoints` |
| `PYTHON` | `python3` |

Model alias overrides:

| Variable | Default model |
|---|---|
| `VTC_MODEL_QWEN2_5_7B` | `Qwen/Qwen2.5-7B-Instruct` |
| `VTC_MODEL_QWEN3_4B` | `Qwen/Qwen3-4B-Instruct-2507` |
| `VTC_MODEL_QWEN3_5_4B` | `Qwen/Qwen3.5-4B` |
| `VTC_MODEL_QWEN3_8B` | `Qwen/Qwen3-8B` |
| `VTC_MODEL_QWEN2_5_VL_7B` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `VTC_MODEL_DEEPSEEK_OCR` | `deepseek-ai/DeepSeek-OCR` |

Each variable may instead point to a local model directory.

The generation scripts intentionally do not score their own answers. To apply
the paper's official per-question-type LongMemEval protocol with the pinned
Llama-3.1-70B-Instruct judge:

```bash
experiments/vtc_memory_validation/docker/run.sh \
  bash scripts/official_judge.sh /runs/results/results_softtoken_simple_f4.json
```

The judge defaults to revision
`1605565b47bb9346c5515c34102e054115b4f98b`, FP8, tensor parallelism 2,
temperature 0, and 10 output tokens. Access to Meta's gated model repository is
required. Set `HF_TOKEN` before starting the container.

## DeepSeek-OCR

DeepSeek-OCR may require a transformers version different from the main
environment. Create its isolated environment with:

```bash
experiments/vtc_memory_validation/scripts/setup_deepseek_ocr.sh
experiments/vtc_memory_validation/scripts/deepseek_ocr.sh
```

Set `DSOCR_PYTHON` to use another prepared environment.

## Outputs

Checkpoints and result JSON files are written to their local directories above.
Soft-token checkpoints include all architecture and training arguments;
`eval_softtoken.py` restores mode, factor, encoder depth, role factors, and
pooling type directly from the checkpoint, so no sidecar JSON config is needed.
