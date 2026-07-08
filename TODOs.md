# Experiment TODOs

## Evidence Pattern to Match

Strong neighboring papers do not rely on one headline curve alone. They usually combine:

- end-task performance
- mechanism-isolation ablations
- efficiency / cost / practicality evidence

Representative references from the literature scan:

- `Gist Tokens`: controls, human eval, failure cases, compute analysis
- `ICAE`: downstream results, scalability, latency, analysis
- `LLMLingua` / `LongLLMLingua`: task breadth, baselines, ablations, latency, cost
- `xRAG`: baseline comparison, efficiency, "what makes it effective" analysis
- `RECOMP`: baselines + oracles + faithfulness/comprehensiveness checks
- `LongMemEval`: QA accuracy plus memory recall
- `DeepSeek-OCR` / `Glyph`: benchmark performance plus efficiency and qualitative validation

## Highest-Priority TODOs

- [ ] Add budget-matched role ablations:
  `user-lossless / assistant-compressed`, `assistant-lossless / user-compressed`, `random preserved turns`, `recency-preserved turns`, `uniform soft-token`
- [ ] Add stronger raw-context controls:
  optimized raw prompt, `user-only raw`, retrieval baseline, and `user + assistant summary`
- [ ] Add simple heuristic baselines at the same budget:
  `keep user raw + drop assistant`, `keep user raw + summarize assistant`, pruning-only assistant baseline
- [ ] Add a direct support-preservation metric:
  whether answer-supporting facts survive compression, split by support in user vs assistant turns
- [ ] Add confidence intervals or paired significance testing on the main deltas

## Role-Awareness Validation

- [ ] Audit answer-support location on LongMemEval:
  what fraction of answers are grounded in user turns vs assistant turns
- [ ] Repeat the same audit on LoCoMo
- [ ] Correlate role asymmetry with observed gains by category and by reader model

## Baseline Sanity Checks

- [ ] Verify the `compressed > raw` claim against stronger raw baselines before presenting raw as an upper bound
- [ ] Add an oracle-style analysis:
  preserve only true supporting turns or true answer-bearing turns to estimate headroom
- [ ] Add a sanity check for the DeepSeek-OCR pipeline on rendered conversation pages so the baseline cannot be dismissed as misconfigured

## Efficiency and Practicality

- [ ] Expand efficiency beyond compression-step timing:
  report end-to-end latency, throughput, and cost under matched batching
- [ ] Report any baseline brittleness or failure modes explicitly, rather than only mentioning them in passing

## Evaluation Coverage

### Model Families

- [ ] Keep `Qwen` as the anchor family, but add `Llama 3.x` as the main non-Qwen validation family
- [ ] If time allows, add a third family such as `Mistral` / `Ministral`
- [ ] Treat `Qwen + Llama` as the minimum credible cross-family story

### Suggested Reader Set

- [ ] Minimum matrix: `Qwen3-1.7B`, `Qwen3-8B`, `Llama-3.1-8B`
- [ ] Optional weak-reader stress test: small `Phi` or `Llama-3.2-3B`

### Training Corpora

- [ ] Keep `UltraChat` as the clean supervised corpus
- [ ] Add one noisier real user-assistant corpus: `WildChat` or `LMSYS-Chat-1M`
- [ ] Optional: add `ShareGPT` if setup cost is low

### Effectiveness Benchmarks

- [ ] Keep `LongMemEval-S` and `LongMemEval-M` as core memory benchmarks
- [ ] Add one more user-assistant memory benchmark: `PerLTQA` or `MemoryBank`
- [ ] Keep `LoCoMo` QA as the multi-turn conversational benchmark
- [ ] Treat `LongBench` / `RULER` as secondary stress tests, not primary evidence

### Efficiency Benchmarks

- [ ] Measure end-to-end latency, not just compressor latency
- [ ] Measure throughput and peak GPU memory under matched batching
- [ ] Report accuracy at fixed latency and fixed memory budget
- [ ] Add a context-length scaling sweep on `LongMemEval` histories, optionally with a synthetic conversational needle setup

### Recommended Minimal Matrix

- [ ] Readers: `Qwen3-1.7B`, `Qwen3-8B`, `Llama-3.1-8B`
- [ ] Training data: `UltraChat` + `WildChat` or `LMSYS-Chat-1M`
- [ ] Evaluation: `LongMemEval-S`, `LongMemEval-M`, `PerLTQA` or `MemoryBank`, `LoCoMo`
- [ ] Efficiency profiling: end-to-end measurements plus context-length sweeps

## Nice-to-Have

- [ ] Add one non-Qwen reader family
- [ ] Add a short qualitative error analysis comparing role-aware vs uniform failures
- [ ] Reframe the paper around the strongest causal claim:
  role asymmetry helps because answer-bearing support is concentrated in user turns
