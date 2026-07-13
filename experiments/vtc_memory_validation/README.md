# VTC-vs-Text on Conversational Memory — Zero-Training Validation

**Goal:** cheaply check the core research hypothesis before writing any training code:

> Visual-Text Compression (VTC) hurts **conversational memory** more than text-based
> compression, especially on **precise-fact / multi-hop** questions — because dialogue
> is low-density and memory QA needs precise retrieval of user-stated facts.

No training. Just compare three ways of feeding a long LoCoMo conversation to a model
and measure memory-QA accuracy **by question category** + the achieved compression ratio:

| condition | how the history is fed | role |
|---|---|---|
| `raw`     | full conversation as text            | upper bound |
| `summary` | LLM-summarized conversation as text  | text compression baseline |
| `vtc`     | conversation rendered to images → VLM | visual compression (the thing we suspect fails) |

**Expected signal (what would confirm the hypothesis):** `vtc` accuracy drops well
below `raw`/`summary`, and the drop is largest on `single_hop` / `multi_hop` /
`temporal` (precise retrieval), smaller on `open_domain`. If instead `vtc` ≈ `summary`
everywhere, the hypothesis is weak and we rethink.

---

## 1. Launch a GPU pod

This repo now ships its own `mldev` config (`openconnect.json` +
`workspace/src/workflows/interactive.py`), so you can launch a pod directly from
here — no dependency on RLPilot.

**Step 1a — launch a pod (from this repo root on your Mac):**

```bash
cd /path/to/mldev_efficiency

# VSCode-in-browser pod (h200_1)
mldev run idev -d prod-lor1-k8s-2 --crew-id 3330
# or an H100_2 pod:
mldev run h100 -d prod-lva1-k8s-2 --crew-id 3330
# available: idev (h200_1 vscode), h100 (h100_2), h100_8, h200 (h200_2)
```

The `experiments/` directory is shipped as a resource, so the code lands on the
pod. Grab the execution ID from the printed Flyte URL, wait for `Running`, then
port-forward to the VSCode server:

```bash
kubectl config use-context prod-lor1-k8s-2
kubectl config set-context --current --namespace=training-coreai
kubectl get pods -n training-coreai | grep <EXEC_ID>
kubectl port-forward <EXEC_ID>-n0-0-master-0 8080:8080 -n training-coreai
# open http://localhost:8080
```

**Step 1b — on the pod, go to the experiment dir:**

```bash
unset HTTPS_PROXY
cd experiments/vtc_memory_validation   # shipped with the pod
```

## 2. Install deps

The `mldev_verl_vllm_cu128_image` already has torch + transformers, but pin the extras:

```bash
pip install -r requirements.txt
```

## 3. Prepare data

For the zero-shot soft-token experiments, training uses a sampled UltraChat subset and evaluation uses LongMemEval:

```bash
python prepare_data.py --stage_dir /shared/public/sharing/vtc_memory/data
```

`prepare_data.py` writes `data/ultrachat_train.json` by reusing the bundled tracked
2,000-conversation UltraChat subset in `longmemeval_evaluation_training_data/`
unless `--refresh_ultrachat` is passed. `--stage_dir` copies this UltraChat subset
and LongMemEval to the NFS location used by offline batch sweeps.

## 4. Smoke test, then full run

```bash
# 1) text path only, 5 items — confirms data loads + text model works
python run_validation.py --limit 5 --conditions raw

# 2) add text compression
python run_validation.py --limit 10 --conditions raw,summary

# 3) full three-way comparison
python run_validation.py --limit 30 --conditions raw,summary,vtc
```

Output: a per-category accuracy table for each condition + mean compression ratio,
and `results.json` with every prediction for inspection.

For the UltraChat-subset → LongMemEval soft-token sweep:

```bash
mldev run vtc_sweep -e softtoken_full_u1_zeroshot -d prod-lva1-k8s-2 --crew-id 3330
```

---

## Things to verify on first run (I could not test locally — this Mac has no GPU)

1. **LoCoMo data URL.** Default is
   `https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json`.
   If it 404s, find the correct `locomo10.json` and pass `--data_path /path/to/locomo10.json`.
   The parser expects each sample to have `conversation` (with `session_N` +
   `session_N_date_time`) and `qa` (with `question`, `answer`, `category`).
2. **Qwen2.5-VL class import.** Needs `transformers>=4.49`. If the
   `Qwen2_5_VLForConditionalGeneration` import fails, upgrade transformers.
3. **Font for rendering.** Falls back to `DejaVuSansMono.ttf` then PIL default.
   Pass `--font_path` if you want a specific font.
4. **Compression ratio knobs.** `--summary_ratio` (text) and `--font_size` (smaller
   font → more text per image → higher VTC ratio) let you match compression ratios
   across conditions for a fair comparison.

## Next steps after the signal

- If confirmed: add **LongMemEval** (primary benchmark, has info-extraction /
  knowledge-update types) and the attribution probes (entity-shuffle to remove
  language priors; user-message vs assistant-message compression sensitivity).
- Then move to the method (memory-aware selective compression).
