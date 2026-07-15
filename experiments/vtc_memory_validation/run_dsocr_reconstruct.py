#!/usr/bin/env python3
"""
Stage 1 of the DeepSeek-OCR visual-compression condition.

Runs in the ISOLATED transformers==4.46.3 venv (DeepSeek-OCR needs the old
Llama attention classes; it conflicts with the main env's transformers 5.x).

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

    # ---- load DeepSeek-OCR ----
    import torch
    from transformers import AutoModel, AutoTokenizer
    model_path = rv.resolve_model(args.model_path)
    print(f"[dsocr] loading {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, _attn_implementation="eager",
    ).eval().cuda().to(torch.bfloat16)

    tmpdir = tempfile.mkdtemp(prefix="dsocr_")
    out_path = os.path.join(tmpdir, "out")
    cache = {}

    def ocr_text(text_block, si, tag):
        """Render a text block to pages, OCR each, return (recon_text, n_pages)."""
        pages = rv.render_text_to_images(
            text_block, font_size=args.font_size,
            page_w=args.render_size, page_h=args.render_size)
        recon = []
        for pi, page in enumerate(pages):
            img_path = os.path.join(tmpdir, f"p_{si}_{tag}_{pi}.png")
            page.save(img_path)
            txt = model.infer(
                tok, prompt="<image>\nFree OCR.", image_file=img_path,
                output_path=out_path, base_size=args.base_size,
                image_size=args.base_size, crop_mode=False, eval_mode=True)
            recon.append(txt if isinstance(txt, str) else str(txt))
        return "\n".join(recon), len(pages)

    for n, (si, conv_text) in enumerate(convs.items()):
        if args.mode == "simple":
            recon, npages = ocr_text(conv_text, si, "all")
            vision_tokens = npages * tokens_per_page
            text_tokens = 0
        else:
            # full: user turns kept verbatim; only ASSISTANT turns OCR-compressed.
            # Consecutive assistant turns are batched into one rendered block.
            turns = _turns_for_sample(args.dataset, data, si)
            parts, npages, text_tokens = [], 0, 0
            asst_buf = []
            for t in turns:
                if t["role"] == "user":
                    if asst_buf:
                        rec, p = ocr_text("\n".join(asst_buf), si, f"a{npages}")
                        parts.append(rec)
                        npages += p
                        asst_buf = []
                    line = f"user: {t['content']}"
                    parts.append(line)
                    text_tokens += len(tok(line)["input_ids"])
                else:
                    asst_buf.append(f"assistant: {t['content']}")
            if asst_buf:
                rec, p = ocr_text("\n".join(asst_buf), si, f"a{npages}")
                parts.append(rec)
                npages += p
            recon = "\n".join(parts)
            vision_tokens = npages * tokens_per_page

        key = f"{args.dataset}:{si}"
        cache[key] = {
            "reconstructed": recon,
            "vision_tokens": vision_tokens,
            "text_tokens": text_tokens,
            "pages": npages,
        }
        if (n + 1) % 5 == 0:
            print(f"  ... {n + 1}/{len(convs)} conversations reconstructed")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "cache": cache}, f)
    print(f"[dsocr] wrote {len(cache)} entries to {args.out}")


if __name__ == "__main__":
    main()
