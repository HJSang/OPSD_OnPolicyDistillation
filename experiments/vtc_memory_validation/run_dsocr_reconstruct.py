#!/usr/bin/env python3
"""
Stage 1 of the DeepSeek-OCR visual-compression condition.

The batched vLLM engine runs in the main pinned environment. The native
Transformers fallback runs in an isolated transformers==4.46.3 environment
because DeepSeek-OCR's remote model code needs the older Llama attention
classes.

For each conversation it:
  1. renders the conversation text to square page images,
  2. OCR-reconstructs each page with DeepSeek-OCR (the visual bottleneck),
  3. records the reconstructed text + the number of vision tokens used.

DeepSeek-OCR vision-token count per page (crop_mode=False) = (base_size/64)^2:
    base_size 512 -> 64 (Tiny), 640 -> 100 (Small),
              1024 -> 256 (Base), 1280 -> 400 (Large).

Output: a cache JSON keyed by conversation id ("<dataset>:<sample_index>"),
consumed by run_validation.py's `dsocr` condition (which answers with Qwen over
the reconstructed text -> apples-to-apples with the `summary` condition).

Usage:
    python run_dsocr_reconstruct.py --dataset locomo --data_path data/locomo10.json \
        --limit 20 --base_size 1024 --font_size 16 \
        --model_path deepseek-ocr --out results/dsocr_cache_locomo_b1024.json
"""
import argparse
import gc
import json
import os
import tempfile

import run_validation as rv  # top-level imports are stdlib + PIL only (safe)


def _turns_for_sample(dataset_name, raw, si):
    """First-class per-turn role structure for a sample (no torch import)."""
    if dataset_name == "longmemeval":
        inst = raw[si]
        return [{"role": t.get("role", "user"), "content": t.get("content", "")}
                for s in inst.get("haystack_sessions", []) for t in s]
    import re
    conv = raw[si]["conversation"]
    sids = sorted(int(k.split("_")[1]) for k in conv
                  if re.fullmatch(r"session_\d+", k))
    turns = []
    for sid in sids:
        for t in conv[f"session_{sid}"]:
            role = "user" if t.get("speaker") == conv.get("speaker_a") else "assistant"
            turns.append({"role": role, "content": t.get("text", "")})
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo", choices=["locomo", "longmemeval"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model_path", default="deepseek-ocr",
                    help="Registry name (e.g. deepseek-ocr) or full path")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--limit_per_sample", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--font_size", type=int, default=16)
    ap.add_argument("--base_size", type=int, default=1024,
                    help="DeepSeek-OCR OCR resolution; tokens/page=(base_size/64)^2. "
                         "Lower -> fewer vision tokens -> more compression (and "
                         "blurrier text) for the SAME rendered page.")
    ap.add_argument("--render_size", type=int, default=1024,
                    help="Pixel size of each square rendered page (kept fixed "
                         "across the base_size sweep so higher compression = "
                         "downsampling a constant-density page).")
    ap.add_argument("--out", default="dsocr_cache.json")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "native"],
                    help="batched vLLM matches the paper; native is a slow "
                         "per-page compatibility fallback")
    ap.add_argument("--ocr_max_tokens", type=int, default=2048)
    ap.add_argument("--gpu_mem_util", type=float, default=0.70)
    ap.add_argument("--max_num_seqs", type=int, default=16)
    ap.add_argument("--max_num_batched_tokens", type=int, default=16384)
    ap.add_argument("--enforce_eager", action="store_true",
                    help="disable CUDA graphs (automatically enabled at 1024)")
    ap.add_argument("--mode", default="simple", choices=["simple", "full"],
                    help="simple=render whole conversation; full=keep USER turns "
                         "as verbatim text and only OCR-compress ASSISTANT turns "
                         "(DeepSeek-OCR analogue of soft-token full_u1).")
    args = ap.parse_args()

    tokens_per_page = (args.base_size // 64) ** 2

    # ---- select the same items run_validation will evaluate ----
    data = rv.load_json(args.data_path)
    items = list(rv.iter_items(args.dataset, data, args.limit_per_sample))
    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(items)
    if args.limit:
        items = items[: args.limit]

    # dedupe conversations by sample index (LoCoMo shares a conv across QAs)
    convs = {}
    for si, conv_text, *_ in items:
        convs.setdefault(si, conv_text)
    print(f"[dsocr] {len(convs)} unique conversations to reconstruct "
          f"(base_size={args.base_size}, {tokens_per_page} tok/page)")

    model_path = rv.resolve_model(args.model_path)
    print(f"[dsocr] loading {model_path}")
    if args.engine == "vllm":
        if args.mode != "simple":
            raise ValueError("the batched vLLM engine currently supports simple mode")
        cache = reconstruct_vllm(args, model_path, convs, tokens_per_page)
    else:
        cache = reconstruct_native(
            args, model_path, data, convs, tokens_per_page)

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "cache": cache}, f)
    print(f"[dsocr] wrote {len(cache)} entries to {args.out}")


def reconstruct_vllm(args, model_path, convs, tokens_per_page):
    """Render all pages and reconstruct them in one vLLM generate call."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models import deepseek_ocr as vllm_dsocr_model
    from vllm.transformers_utils.processors import deepseek_ocr as vllm_dsocr_processor

    # vLLM 0.11.2 exposes these modes as module constants rather than processor
    # kwargs. Both modules import their own copies, so update them before LLM
    # construction and let the processor use its default call signature.
    for module in (vllm_dsocr_model, vllm_dsocr_processor):
        module.BASE_SIZE = args.base_size
        module.IMAGE_SIZE = args.base_size
        module.CROP_MODE = False
    ngram_processor = vllm_dsocr_model.NGramPerReqLogitsProcessor

    def override_hf_config(config):
        config.architectures = ["DeepseekOCRForCausalLM"]
        vision_config = config.vision_config
        if isinstance(vision_config, dict):
            vision_config["image_size"] = args.base_size
        else:
            vision_config.image_size = args.base_size
        return config

    prompts = []
    owners = []
    pages_per_sample = {}
    for n, (si, conv_text) in enumerate(convs.items()):
        pages = rv.render_text_to_images(
            conv_text, font_size=args.font_size,
            page_w=args.render_size, page_h=args.render_size)
        pages_per_sample[si] = len(pages)
        for page in pages:
            prompts.append({
                "prompt": "<image>\nFree OCR.",
                "multi_modal_data": {
                    "image": page.resize(
                        (args.base_size, args.base_size), Image.Resampling.BILINEAR)
                },
            })
            owners.append(si)
        if (n + 1) % 25 == 0:
            print(f"  ... rendered {n + 1}/{len(convs)} conversations")
    print(f"[dsocr] batched OCR pages={len(prompts)}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=8192,
        block_size=256,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="auto",
        enforce_eager=args.enforce_eager or args.base_size == 1024,
        mm_processor_cache_gb=0,
        logits_processors=[ngram_processor],
        hf_overrides=override_hf_config,
    )
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.ocr_max_tokens,
        skip_special_tokens=False,
        extra_args={
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": {128821, 128822},
        },
    )
    outputs = llm.generate(prompts, sampling)
    reconstructed = {si: [] for si in convs}
    for si, output in zip(owners, outputs):
        reconstructed[si].append(
            output.outputs[0].text if output.outputs else "")

    cache = {}
    for si, texts in reconstructed.items():
        npages = pages_per_sample[si]
        cache[f"{args.dataset}:{si}"] = {
            "reconstructed": "\n".join(texts),
            "vision_tokens": npages * tokens_per_page,
            "text_tokens": 0,
            "pages": npages,
        }
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return cache


def reconstruct_native(args, model_path, data, convs, tokens_per_page):
    """Compatibility path using DeepSeek-OCR's sequential model.infer API."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, _attn_implementation="eager",
    ).eval().cuda().to(torch.bfloat16)
    tmpdir = tempfile.mkdtemp(prefix="dsocr_")
    out_path = os.path.join(tmpdir, "out")
    cache = {}

    def ocr_text(text_block, si, tag):
        pages = rv.render_text_to_images(
            text_block, font_size=args.font_size,
            page_w=args.render_size, page_h=args.render_size)
        recon = []
        for pi, page in enumerate(pages):
            img_path = os.path.join(tmpdir, f"p_{si}_{tag}_{pi}.png")
            page.save(img_path)
            text = model.infer(
                tok, prompt="<image>\nFree OCR.", image_file=img_path,
                output_path=out_path, base_size=args.base_size,
                image_size=args.base_size, crop_mode=False, eval_mode=True)
            recon.append(text if isinstance(text, str) else str(text))
        return "\n".join(recon), len(pages)

    for n, (si, conv_text) in enumerate(convs.items()):
        if args.mode == "simple":
            recon, npages = ocr_text(conv_text, si, "all")
            text_tokens = 0
        else:
            turns = _turns_for_sample(args.dataset, data, si)
            parts, npages, text_tokens = [], 0, 0
            assistant_buffer = []
            for turn in turns:
                if turn["role"] == "user":
                    if assistant_buffer:
                        text, count = ocr_text(
                            "\n".join(assistant_buffer), si, f"a{npages}")
                        parts.append(text)
                        npages += count
                        assistant_buffer = []
                    line = f"user: {turn['content']}"
                    parts.append(line)
                    text_tokens += len(tok(line)["input_ids"])
                else:
                    assistant_buffer.append(f"assistant: {turn['content']}")
            if assistant_buffer:
                text, count = ocr_text(
                    "\n".join(assistant_buffer), si, f"a{npages}")
                parts.append(text)
                npages += count
            recon = "\n".join(parts)
        cache[f"{args.dataset}:{si}"] = {
            "reconstructed": recon,
            "vision_tokens": npages * tokens_per_page,
            "text_tokens": text_tokens,
            "pages": npages,
        }
        if (n + 1) % 5 == 0:
            print(f"  ... {n + 1}/{len(convs)} conversations reconstructed")
    return cache


if __name__ == "__main__":
    main()
