# LongMemEval Reproduction Results

## Corrected Chat-Template Run

The current reader protocol applies the Qwen2.5-Instruct chat template to both
text-token and soft-token inputs. Continuous context embeddings replace a
placeholder inside the rendered user turn, before the assistant-generation
prefix.

One complete seed-0 run was verified on all 500 LongMemEval oracle-evidence
questions:

| Method | Compression | Overall | User-fact |
|---|---:|---:|---:|
| Paper: SoftMem `user=1, assistant=8` (5 seeds) | 4.62x | 0.476 +/- 0.014 | 0.938 +/- 0.007 |
| Reproduced: SoftMem `user=1, assistant=8` (seed 0) | 4.6163x | 0.512 | 0.9531 |

The official per-question-type judge was
`meta-llama/Meta-Llama-3.1-70B-Instruct` at revision
`1605565b47bb9346c5515c34102e054115b4f98b`, with greedy decoding and FP8
inference. All 500 judge responses were canonical yes/no answers.

This run verifies one training seed and the complete public reproduction
pipeline. It does not reproduce the paper's five-seed variance.

## Tables 2 And 3: Base Versus Instruct

The paper abbreviates its reader as Qwen2.5-7B but specifies
`Qwen/Qwen2.5-7B-Instruct` in the setup. A controlled one-seed run evaluated
both that checkpoint and `Qwen/Qwen2.5-7B`, with each checkpoint's chat
template.

### Table 2: LongMemEval

| Method | Paper compression | Reproduced compression | Paper overall | Instruct reproduced | Delta |
|---|---:|---:|---:|---:|---:|
| SoftMem `a8` | 4.62x | 4.6163x | 0.476 | 0.512 | +0.036 |
| SoftMem `a16` | 6.27x | 6.2731x | 0.478 | 0.502 | +0.024 |
| SoftMem `a32` | 7.70x | 7.6962x | 0.504 | 0.498 | -0.006 |
| DeepSeek-OCR `b1024` | 2.33x | 2.2605x | 0.426 | 0.438 | +0.012 |
| DeepSeek-OCR `b640` | 5.97x | 5.7869x | 0.390 | 0.412 | +0.022 |
| DeepSeek-OCR `b512` | 9.34x | 9.0420x | 0.312 | 0.380 | +0.068 |
| Text summary | 12.06x | 19.9328x | 0.184 | 0.170 | -0.014 |

The base-model control produced `0.370/0.380/0.372` for SoftMem,
`0.206/0.198/0.190` for DeepSeek-OCR, and `0.092` for text summary. This
confirms that the paper row is an Instruct-model result.

### Table 3: Raw LongBench-QA

| Metric | Paper | Instruct reproduced | Delta |
|---|---:|---:|---:|
| Overall | 43.80 | 41.88 | -1.92 |
| narrativeqa | 29.29 | 27.85 | -1.44 |
| qasper | 44.14 | 40.45 | -3.69 |
| multifieldqa_en | 52.32 | 52.18 | -0.14 |
| hotpotqa | 58.40 | 56.98 | -1.42 |
| 2wikimqa | 47.80 | 44.68 | -3.12 |
| musique | 30.85 | 29.14 | -1.71 |

The base-model control scored `5.35` overall. Every Table 2 method was judged
over 500 unique question IDs, and every Table 3 score used the full official
subset.

## Legacy Full-Table Sweep

The earlier full-table sweep used bare text continuations instead of the
Instruct model's chat template. Those results remain useful as historical
pipeline validation, but they are a different prompt protocol and should not be
combined with the corrected result above.

| Method | Compression | Overall |
|---|---:|---:|
| Raw | 1.000x | 0.442 |
| Role-aware `assistant=8` | 4.616x | 0.462 |
| Role-aware `assistant=16` | 6.273x | 0.466 |
| Role-aware `assistant=32` | 7.696x | 0.478 |
| Uniform factor 4 | 3.998x | 0.114 |
| Uniform factor 8 | 7.991x | 0.112 |
| Uniform factor 16 | 15.962x | 0.180 |
| DeepSeek-OCR base 1024 | 2.261x | 0.438 |
| DeepSeek-OCR base 640 | 5.787x | 0.412 |
| DeepSeek-OCR base 512 | 9.042x | 0.380 |
| Text summary | 19.933x | 0.170 |

All current runs record their resolved prompt format in result metadata.
Checkpoints created by the corrected code also record `prompt_format=chat`;
older checkpoints default to `plain` during evaluation.

See [README.md](README.md) for the one-result and full-table reproduction
commands.
