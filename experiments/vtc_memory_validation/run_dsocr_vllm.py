#!/usr/bin/env python3
"""
vLLM-batched DeepSeek-OCR reconstruction (fast path, ~25x over transformers).

Renders each conversation to page images (fixed render size), OCR-reconstructs
ALL pages in one batched vLLM call, and writes the same cache format as
run_dsocr_reconstruct.py so run_validation.py's `dsocr` condition can consume it.

Run in the vLLM venv (transformers pinned by vllm; DeepSeek-OCR arch needs
vllm>=0.11.2):
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
    ~/vllm_env/bin/python run_dsocr_vllm.py --dataset longmemeval \
        --data_path data/longmemeval_oracle.json --limit 100 --shuffle --seed 0 \
        --base_size 640 --out dsocr_cache_lme100_b640.json
"""
import argparse
import json
import os

import run_validation as rv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="longmemeval",
                    choices=["locomo", "longmemeval"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model_path",
                    default=os.environ.get(
                        "VTC_DEEPSEEK_OCR",
                        "/shared/public/sharing/vtc_memory/DeepSeek-OCR"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--limit_per_sample", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--font_size", type=int, default=16)
    ap.add_argument("--base_size", type=int, default=640,
                    help="OCR resolution; vision tokens/page=(base_size/64)^2")
    ap.add_argument("--render_size", type=int, default=1024)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--out", default="dsocr_cache_vllm.json")
    args = ap.parse_args()

    tokens_per_page = (args.base_size // 64) ** 2

    # select the same items run_validation will evaluate
    data = rv.load_json(args.data_path)
    items = list(rv.iter_items(args.dataset, data, args.limit_per_sample))
    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(items)
    if args.limit:
        items = items[: args.limit]

    # dedupe conversations, render all pages, remember page->conversation mapping
    convs = {}
    for si, conv_text, *_ in items:
        convs.setdefault(si, conv_text)
    print(f"[dsocr-vllm] {len(convs)} conversations, base_size={args.base_size} "
          f"({tokens_per_page} tok/page)")

    # Render every page, resized to the OCR base_size (vLLM path takes the image
    # as-is, so we downsample here to control the vision-token budget).
    from PIL import Image
    all_prompts = []        # one entry per page
    page_owner = []         # si for each page
    pages_per_conv = {}
    for si, conv_text in convs.items():
        pages = rv.render_text_to_images(
            conv_text, font_size=args.font_size,
            page_w=args.render_size, page_h=args.render_size)
        pages_per_conv[si] = len(pages)
        for pg in pages:
            img = pg.resize((args.base_size, args.base_size), Image.BILINEAR)
            all_prompts.append({
                "prompt": "<image>\nFree OCR.",
                "multi_modal_data": {"image": img},
            })
            page_owner.append(si)
    print(f"[dsocr-vllm] {len(all_prompts)} total pages to OCR (batched)")

    # load vLLM
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_path, trust_remote_code=True, max_model_len=8192,
              gpu_memory_utilization=args.gpu_mem_util,
              hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]})
    sp = SamplingParams(temperature=0.0, max_tokens=2048)

    # ONE batched call over all pages
    outputs = llm.generate(all_prompts, sp)

    # regroup page texts by conversation
    recon_by_conv = {si: [] for si in convs}
    for owner, out in zip(page_owner, outputs):
        txt = out.outputs[0].text if out.outputs else ""
        recon_by_conv[owner].append(txt)

    cache = {}
    for si in convs:
        npages = pages_per_conv[si]
        cache[f"{args.dataset}:{si}"] = {
            "reconstructed": "\n".join(recon_by_conv[si]),
            "vision_tokens": npages * tokens_per_page,
            "text_tokens": 0,
            "pages": npages,
        }
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "cache": cache}, f)
    print(f"[dsocr-vllm] wrote {len(cache)} entries to {args.out}")


if __name__ == "__main__":
    main()
