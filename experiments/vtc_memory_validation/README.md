# Reproducing Conversational-Memory Compression

This directory contains the training, evaluation, and judging code for the
paper's conversational-memory experiments. The primary benchmark is
LongMemEval with oracle evidence. The evaluated context representations are:

| Method | Context representation |
|---|---|
| `raw` | Original text |
| `summary` | Model-generated factual summary |
| `dsocr` | Text rendered as images and reconstructed by DeepSeek-OCR |
| `softtoken simple` | Uniform learned pooling |
| `softtoken role-aware` | User tokens preserved; assistant tokens pooled |

The supported reproduction path uses only Docker, NVIDIA's container runtime,
public container images, public datasets, and public Hugging Face model
repositories. It does not require a cluster scheduler, shared filesystem, or
organization-specific service.

## Prerequisites

- Linux on x86-64
- Docker Engine with the NVIDIA Container Toolkit
- An NVIDIA GPU and a driver compatible with CUDA 12.9
- Sufficient local storage for the selected model checkpoints
- A Hugging Face token with access to
  `meta-llama/Meta-Llama-3.1-70B-Instruct` for official judging

Training and generation for one SoftMem configuration use one GPU. The default
70B FP8 judge uses two GPUs. The full-table launcher runs the eight generation
jobs concurrently and therefore expects eight GPUs.

The environment was validated on NVIDIA B200 GPUs. Other recent NVIDIA GPUs
should work when they have enough memory; lower batch and concurrency settings
where noted below.

## Build The Environment

Run all commands from the repository root:

```bash
experiments/vtc_memory_validation/docker/build.sh

experiments/vtc_memory_validation/docker/run.sh \
  python docker/verify_environment.py --require-gpu
```

The image extends the public, digest-pinned
`slimerl/slime@sha256:9b548a94930f5b1b03faaf481d0ed5c31d12302b7ca37cf2ca933c9c60d0975e`
image. The verifier checks the Python package versions, imports the actual vLLM
judge API, resolves the mapped container user, and executes a CUDA operation.

Key versions are:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| CUDA | 12.9 |
| cuDNN | 9.16.0 |
| NCCL | 2.27.5 |
| PyTorch | 2.9.1+cu129 |
| Transformers | 4.57.1 |
| SGLang | 0.5.9 |
| Ray | 2.54.0 |
| vLLM | 0.11.2 |

By default, Hugging Face files are cached under `~/.cache/huggingface` and run
artifacts are stored under
`experiments/vtc_memory_validation/.docker-runs`. Override these locations with
`VTC_HF_HOME` and `VTC_RUNS_DIR`.

## Reproduce One Main Result

The following commands train and evaluate the role-aware `user=1,
assistant=8` configuration used in the first SoftMem row of the paper's main
LongMemEval table. They retain the checkpoint, predictions, official scores,
and complete logs.

```bash
export HF_TOKEN=your_hugging_face_token
export VTC_RUNS_DIR="$PWD/experiments/vtc_memory_validation/.docker-runs/softmem-a8"

experiments/vtc_memory_validation/docker/run.sh \
  python prepare_data.py --data_dir /runs/data \
  2>&1 | tee "$VTC_RUNS_DIR/logs/prepare-data.log"

CUDA_VISIBLE_DEVICES=0 \
experiments/vtc_memory_validation/docker/run.sh \
  bash -lc 'ASSISTANT_FACTORS=8 LIMIT=500 bash scripts/softtoken_role_aware.sh' \
  2>&1 | tee "$VTC_RUNS_DIR/logs/softmem-a8.log"

CUDA_VISIBLE_DEVICES=0,1 \
experiments/vtc_memory_validation/docker/run.sh \
  bash scripts/official_judge.sh \
    /runs/results/results_softtoken_u1_a8.json \
  2>&1 | tee "$VTC_RUNS_DIR/logs/official-judge-a8.log"
```

The verified seed-0 run produced:

| Metric | Paper, 5 seeds | Reproduced, seed 0 |
|---|---:|---:|
| Compression | 4.62x | 4.6163x |
| Overall accuracy | 0.476 +/- 0.014 | 0.512 |
| User-fact accuracy | 0.938 +/- 0.007 | 0.9531 |

A single run verifies the pipeline and one seed; it does not reproduce the
paper's five-seed mean or standard deviation.

The main artifacts are:

```text
$VTC_RUNS_DIR/checkpoints/softtoken_u1_a8.pt
$VTC_RUNS_DIR/results/results_softtoken_u1_a8.json
$VTC_RUNS_DIR/results/results_softtoken_u1_a8_official_judge.json
$VTC_RUNS_DIR/logs/
```

## Reproduce The Full Main Table

After building the image and exporting `HF_TOKEN`, run:

```bash
experiments/vtc_memory_validation/scripts/reproduce_main_table.sh
```

The script:

1. Verifies the container environment.
2. Downloads pinned model revisions.
3. Prepares the public datasets.
4. Runs raw, summary, three uniform SoftMem, three role-aware SoftMem, and three
   DeepSeek-OCR configurations across eight GPUs.
5. Loads the official 70B judge once and scores all generated predictions.
6. Retains a source manifest and one log per stage.

The default output directory is
`experiments/vtc_memory_validation/.docker-runs/main-table`.

When running an exported source archive instead of a Git checkout, set
`VTC_GIT_COMMIT` to the source revision that produced the archive.

## Models And Data

The full-table launcher pins these model revisions:

| Purpose | Repository | Revision |
|---|---|---|
| SoftMem writer and reader | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Visual reconstruction | `deepseek-ai/DeepSeek-OCR` | `9f30c71f441d010e5429c532364a86705536c53a` |
| Official judge | `meta-llama/Meta-Llama-3.1-70B-Instruct` | `1605565b47bb9346c5515c34102e054115b4f98b` |

`prepare_data.py` copies the tracked UltraChat subset and downloads LoCoMo and
LongMemEval from their public sources. To prepare data outside Docker:

```bash
cd experiments/vtc_memory_validation
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 prepare_data.py
```

Model aliases may point to another public repository or a local directory:

| Variable | Default |
|---|---|
| `VTC_MODEL_QWEN2_5_7B` | `Qwen/Qwen2.5-7B-Instruct` |
| `VTC_MODEL_QWEN3_4B` | `Qwen/Qwen3-4B-Instruct-2507` |
| `VTC_MODEL_QWEN3_5_4B` | `Qwen/Qwen3.5-4B` |
| `VTC_MODEL_QWEN3_8B` | `Qwen/Qwen3-8B` |
| `VTC_MODEL_QWEN2_5_VL_7B` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `VTC_MODEL_DEEPSEEK_OCR` | `deepseek-ai/DeepSeek-OCR` |

## Run Individual Methods

The entrypoints can be invoked from any working directory:

```bash
# Raw text and summary baselines
experiments/vtc_memory_validation/scripts/raw_summary.sh

# Uniform SoftMem factors 4, 8, and 16
experiments/vtc_memory_validation/scripts/softtoken_simple.sh

# Role-aware SoftMem with user factor 1 and assistant factors 8, 16, and 32
experiments/vtc_memory_validation/scripts/softtoken_role_aware.sh

# DeepSeek-OCR reconstruction
experiments/vtc_memory_validation/scripts/deepseek_ocr.sh

# LongBench task-adapted compressor and raw baseline
experiments/vtc_memory_validation/scripts/longbench.sh
experiments/vtc_memory_validation/scripts/longbench_raw.sh
```

Use environment variables to change a run without editing source files:

```bash
DECODER=Qwen/Qwen2.5-7B-Instruct \
FACTORS="4 8" \
STEPS=200 \
LIMIT=50 \
VTC_DATA_DIR=/path/to/data \
VTC_RESULTS_DIR=/path/to/results \
VTC_CHECKPOINT_DIR=/path/to/checkpoints \
experiments/vtc_memory_validation/scripts/softtoken_simple.sh
```

The main-table SoftMem recipe uses 1,000 steps, 400 UltraChat chunks,
`MAX_LEN=2048`, `BATCH_SIZE=1`, two borrowed encoder layers, mean pooling,
forward-KL weight 1.0, learning rate `1e-4`, and seed 0. Deterministic
PyTorch/CUDA algorithms are enabled by default.

## Prompt Protocol

Reader-facing QA uses the selected Instruct model's chat template, including
its assistant-generation prefix. For soft-token inference, the template is
rendered once around a placeholder and the continuous context embeddings replace
that placeholder. Qwen3 templates are rendered with thinking disabled.

New checkpoints record `prompt_format=chat`. Evaluation in `auto` mode treats
older checkpoints without this field as `plain`, preserving their historical
behavior. Scores from the legacy plain-prompt protocol must not be compared
directly with scores from the chat-template protocol.

## Resource Tuning

The official judge defaults to FP8 and tensor parallelism 2. Set `JUDGE_TP`,
`JUDGE_QUANTIZATION`, or `JUDGE_MODEL` to change it.

The full-table launcher sets DeepSeek-OCR `--max_num_seqs 64`, which was
validated on a 192 GB GPU. Reduce this value on smaller GPUs. The native
DeepSeek-OCR fallback uses the isolated
`/opt/vtc-deepseek-ocr/bin/python` environment included in the image.

Generated data, checkpoints, results, and logs are intentionally excluded from
Git. Keep the complete run directory when reporting a reproduction.
