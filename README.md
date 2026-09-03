# LatentPress

<p align="left">
  <a href="https://arxiv.org/abs/2609.01507"><img src="https://img.shields.io/badge/arXiv-2609.01507-b31b1b.svg" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2609.01507"><img src="https://img.shields.io/badge/paper-PDF-blue.svg" alt="PDF"></a>
  <img src="https://img.shields.io/badge/license-see%20repo-lightgrey.svg" alt="License">
</p>

**Context Compression Beyond Text and Vision**
Zhengze Zhou¹\*, Hejian Sang²\*
¹Cornell University  ²Arizona State University

Compressed context is usually carried as human-readable text or as a rendered
image that must be decoded, even when its only consumer is a language model.
**LatentPress** writes conversational histories and long documents into a third
representation instead: continuous memory tokens that a *frozen* decoder reads
directly through its input-embedding interface, with no text reconstruction at
inference time. A small reader-matched writer (an adapter, not the decoder)
compresses 4–16x while training only ~0.1% of the decoder's parameters
(4.2M–26.2M).

On LongMemEval, LatentPress reaches **0.504 accuracy at 7.70x compression**,
against 0.490 for uncompressed evidence, and beats text summaries (0.184) and
OCR-based compression (0.426 → 0.312 as compression increases). Writing takes
43 ms per conversation — roughly an order of magnitude faster than
summarization or OCR — and reading a compressed prefix is 5–9x faster than
reading raw context or cached OCR.

## Interface

LatentPress separates context use into two operations:

- **WRITE** — maps text (or a document) to a compact continuous state.
- **READ** — supplies that state directly to a frozen decoder's
  input-embedding layer for downstream QA. No decoding back to text.

This differs from the closest prior mechanisms in what is trained, at what
scale, and whether the representation is reconstructed before the decoder
reads it:

| Method | What is trained | Trainable scale | Representation | Reconstructed at inference? |
|---|---|---|---|---|
| [Gist](https://github.com/jayelm/gisting) (Mu et al., 2023) | whole decoder (FT, masked attn.) | decoder-scale | KV-cache | no |
| [AutoCompressor](https://github.com/princeton-nlp/AutoCompressors) (Chevalier et al., 2023) | LLM (recursive summary) | LLM-scale | input (summary) | no |
| [ICAE](https://github.com/getao/icae) (Ge et al., 2024) | LLM encoder (LoRA) | LLM-scale (LoRA) | input slots | yes (autoencoder) |
| [xRAG](https://github.com/Hannibal046/xRAG) (Cheng et al., 2024) | projector only (LLM frozen) | small projector | input (1 token) | no |
| [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) (Wei et al., 2025) | vision model | vision-model-scale | image → text | yes (OCR) |
| [Glyph](https://github.com/thu-coai/Glyph) (Cheng et al., 2025) | vision-text model | vision-model-scale | image → text | yes (OCR) |
| [AgentOCR](https://github.com/langfengQ/AgentOCR) (Feng et al., 2026) | RL-driven visual compression | vision-model-scale | image → text | yes (OCR) |
| **LatentPress (ours)** | small reader-matched adapter | **~0.1% of decoder** | input (soft tokens) | **no** |

Local reference clones of each baseline above live as sibling directories next
to this repo (`../gisting`, `../AutoCompressors`, `../icae`, `../xRAG`,
`../DeepSeek-OCR`, `../Glyph`, `../AgentOCR`) for side-by-side comparison.

## Evaluated Context Representations

The main experiments compare five ways of getting conversational context in
front of a frozen reader model:

| Method | Context representation |
|---|---|
| `raw` | Original text |
| `summary` | Model-generated factual summary |
| `dsocr` | Text rendered as images, reconstructed by DeepSeek-OCR |
| `softtoken simple` | Uniform learned pooling |
| `softtoken role-aware` | User tokens preserved; assistant tokens pooled |

Benchmarks: **LongMemEval** (oracle-evidence conversational memory QA) and
**LongBench-QA** (long-document QA, cross-domain and in-domain adapted).

## Quickstart

Full reproduction instructions (Docker build, pinned model revisions, one main
result, Tables 2/3, and the full main table) are in
[`experiments/vtc_memory_validation/README.md`](experiments/vtc_memory_validation/README.md).

```bash
experiments/vtc_memory_validation/docker/build.sh
experiments/vtc_memory_validation/docker/run.sh \
  python docker/verify_environment.py --require-gpu
```

The repository is self-contained at the orchestration level: experiments are
plain Bash scripts, paths are detected from each script's location, missing
public benchmark data is downloaded automatically, and model aliases resolve
to public Hugging Face repositories.

## Related Repositories

This repo is also referenced as prior/related work from
[`HJSang/OPSD_OnPolicyDistillation`](https://github.com/HJSang/OPSD_OnPolicyDistillation)
(on-policy distillation training, a separate research line by one of the
authors) — see the `latentpress-context-compression` branch there and the
"Related Work" section of its README.

## Citation

```bibtex
@article{zhou2026latentpress,
  title   = {LatentPress: Context Compression Beyond Text and Vision},
  author  = {Zhou, Zhengze and Sang, Hejian},
  journal = {arXiv preprint arXiv:2609.01507},
  year    = {2026}
}
```

## Acknowledgements

Baselines and comparisons build on public releases of
[Gist Tokens](https://github.com/jayelm/gisting),
[AutoCompressors](https://github.com/princeton-nlp/AutoCompressors),
[ICAE](https://github.com/getao/icae),
[xRAG](https://github.com/Hannibal046/xRAG),
[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR),
[Glyph](https://github.com/thu-coai/Glyph), and
[AgentOCR](https://github.com/langfengQ/AgentOCR).
Evaluation uses the public **LongMemEval** and **LongBench-QA** benchmarks and
the **UltraChat** dataset.
