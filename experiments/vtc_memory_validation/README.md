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
