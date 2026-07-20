#!/usr/bin/env python3
"""Generate LongBench-QA predictions through DeepSeek-OCR reconstruction.

Pipeline:
  context -> rendered pages -> DeepSeek-OCR/vLLM reconstructed text
  reconstructed context + official LongBench question prompt -> decoder answer

The prediction files match the official LongBench `eval.py` format.
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
LONGBENCH_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_validation as rv
from softtoken.prompting import render_user_prompt

QA_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def selected_rows(args):
    rows = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        data = list(iter_jsonl(Path(args.data_dir) / f"{dataset}.jsonl"))
        if args.limit:
            data = data[:args.limit]
        for i, row in enumerate(data):
            rows.append((dataset, i, row))
    return rows


def tokens_per_page(base_size):
    base_queries = (base_size // 16 + 3) // 4
    return base_queries * (base_queries + 1) + 1


def build_ocr_cache(args):
    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

    rows = selected_rows(args)
    tok_per_page = tokens_per_page(args.base_size)
    image_size = args.image_size or args.base_size
    all_prompts = []
    owners = []
    pages_per_key = {}

    print(f"[lb-dsocr] rendering {len(rows)} LongBench contexts "
          f"base_size={args.base_size} ({tok_per_page} tok/page)")
    for dataset, idx, row in tqdm(rows):
        key = f"{dataset}:{idx}"
        pages = rv.render_text_to_images(
            row["context"], font_size=args.font_size,
            page_w=args.render_size, page_h=args.render_size)
        pages_per_key[key] = len(pages)
        for page in pages:
            img = page.resize((args.base_size, args.base_size), Image.BILINEAR)
            all_prompts.append({
                "prompt": "<image>\nFree OCR.",
                "multi_modal_data": {"image": img},
            })
            owners.append(key)
    print(f"[lb-dsocr] OCR pages={len(all_prompts)}")

    llm_kwargs = dict(
        model=args.dsocr_model_path,
        trust_remote_code=True,
        max_model_len=8192,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    )
    if args.attention_backend:
        llm_kwargs["attention_config"] = {"backend": args.attention_backend}
    if args.pass_mm_processor_kwargs:
        llm_kwargs["mm_processor_kwargs"] = {
            "base_size": args.base_size,
            "image_size": image_size,
            "crop_mode": args.crop_mode,
        }
    if args.disable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = False
    if args.disable_chunked_prefill:
        llm_kwargs["enable_chunked_prefill"] = False
    if args.disable_async_scheduling:
        llm_kwargs["async_scheduling"] = False

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.ocr_max_tokens,
        skip_special_tokens=False,
        extra_args={
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": {128821, 128822},
        },
    )
    outputs = llm.generate(all_prompts, sp)

    recon = {key: [] for key in pages_per_key}
    for key, out in zip(owners, outputs):
        recon[key].append(out.outputs[0].text if out.outputs else "")
    cache = {}
    for key, texts in recon.items():
        npages = pages_per_key[key]
        cache[key] = {
            "reconstructed": "\n".join(texts),
            "vision_tokens": npages * tok_per_page,
            "pages": npages,
        }

    out_path = Path(args.cache)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "cache": cache}, f, ensure_ascii=False)
    print(f"[lb-dsocr] wrote cache entries={len(cache)} to {out_path}")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_text(decoder, tok, prompt, max_new_tokens, prompt_format):
    if prompt_format == "chat":
        prompt = render_user_prompt(tok, prompt)
    ids = tok(prompt, add_special_tokens=(prompt_format == "plain"),
              truncation=False,
              return_tensors="pt").input_ids.to(decoder.device)
    context_length = ids.shape[-1]
    with torch.no_grad():
        out = decoder.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )[0]
    return tok.decode(out[context_length:], skip_special_tokens=True).strip()


def write_predictions(args):
    official_dir = Path(args.official_dir)
    prompts = load_json(official_dir / "config" / "dataset2prompt.json")
    maxlens = load_json(official_dir / "config" / "dataset2maxlen.json")
    cache = load_json(args.cache)["cache"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = rv.resolve_model(args.decoder)
    print(f"[lb-dsocr] loading decoder {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    decoder = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    out_root = Path(args.out_root) if args.out_root else official_dir / "pred" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {"condition": "dsocr", "decoder": args.decoder,
            "cache": args.cache, "prompt_format": args.prompt_format,
            "datasets": [], "records": {}}

    rows_by_dataset = {}
    for dataset, idx, row in selected_rows(args):
        rows_by_dataset.setdefault(dataset, []).append((idx, row))

    for dataset, rows in rows_by_dataset.items():
        ratios = []
        out_path = out_root / f"{dataset}.jsonl"
        print(f"[lb-dsocr] {dataset}: {len(rows)} predictions -> {out_path}")
        with out_path.open("w", encoding="utf-8") as f:
            for idx, row in tqdm(rows):
                key = f"{dataset}:{idx}"
                cached = cache[key]
                prompt = prompts[dataset].format(
                    context=cached["reconstructed"], input=row["input"])
                pred = generate_text(
                    decoder, tok, prompt, int(maxlens[dataset]),
                    args.prompt_format)
                raw_tokens = len(tok(row["context"], add_special_tokens=False).input_ids)
                ratios.append(raw_tokens / max(1, cached.get("vision_tokens", 0)))
                f.write(json.dumps({
                    "pred": pred,
                    "answers": row["answers"],
                    "all_classes": row.get("all_classes"),
                    "length": row.get("length"),
                    "_id": row.get("_id"),
                }, ensure_ascii=False) + "\n")
        meta["datasets"].append(dataset)
        meta["records"][dataset] = {
            "examples": len(rows),
            "mean_compression_ratio": sum(ratios) / len(ratios) if ratios else None,
            "pages": sum(cache[f"{dataset}:{idx}"]["pages"] for idx, _ in rows),
            "vision_tokens": sum(cache[f"{dataset}:{idx}"]["vision_tokens"] for idx, _ in rows),
        }
    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[lb-dsocr] wrote predictions to {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "pred", "both"], default="both")
    ap.add_argument("--datasets", default=",".join(QA_DATASETS))
    ap.add_argument("--data_dir", default=str(LONGBENCH_ROOT / "data"))
    ap.add_argument("--official_dir", default=str(LONGBENCH_ROOT / "official_eval"))
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out_root", default=None)

    ap.add_argument("--decoder", default="qwen3-8b")
    ap.add_argument("--prompt_format", choices=["chat", "plain"], default="chat",
                    help="Reader prompt format. plain reproduces legacy runs.")
    ap.add_argument("--dsocr_model_path",
                    default=os.environ.get(
                        "VTC_MODEL_DEEPSEEK_OCR",
                        rv.resolve_model("deepseek-ocr")))
    ap.add_argument("--font_size", type=int, default=18)
    ap.add_argument("--base_size", type=int, default=640)
    ap.add_argument("--image_size", type=int, default=None)
    ap.add_argument("--crop_mode", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--pass_mm_processor_kwargs", action="store_true")
    ap.add_argument("--render_size", type=int, default=1024)
    ap.add_argument("--ocr_max_tokens", type=int, default=2048)
    ap.add_argument("--gpu_mem_util", type=float, default=0.70)
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "half", "float16", "bfloat16", "float", "float32"])
    ap.add_argument("--attention_backend", default="FLASHINFER",
                    choices=["FLASH_ATTN", "TRITON_ATTN", "TORCH_SDPA",
                             "FLASHINFER", "FLEX_ATTENTION"])
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--max_num_seqs", type=int, default=16)
    ap.add_argument("--max_num_batched_tokens", type=int, default=16384)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--disable_prefix_caching", action="store_true")
    ap.add_argument("--disable_chunked_prefill", action="store_true")
    ap.add_argument("--disable_async_scheduling", action="store_true")
    args = ap.parse_args()

    if args.stage in ("cache", "both"):
        build_ocr_cache(args)
    if args.stage in ("pred", "both"):
        write_predictions(args)


if __name__ == "__main__":
    main()
