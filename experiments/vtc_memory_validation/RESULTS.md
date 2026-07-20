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
