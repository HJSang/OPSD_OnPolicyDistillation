#!/usr/bin/env python3
"""Generate LongBench-QA predictions from raw text or soft-token memory.

This is an adapter around the official LongBench data/prompt format. It writes
prediction JSONL files that the official `official_eval/eval.py` can score.
"""
import argparse
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
from softtoken.compressor import SoftTokenCompressor

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


def load_softtoken(args, decoder):
    ck = torch.load(args.softtoken_ckpt, map_location="cpu")
    ck_args = ck.get("args", {}) if isinstance(ck, dict) else {}
    train_encoder = "enc_layers" in ck if isinstance(ck, dict) else True
    if isinstance(ck_args, dict) and "train_encoder" in ck_args:
        train_encoder = bool(ck_args["train_encoder"])
    pool_mode = ck_args.get("pool") if isinstance(ck_args, dict) else None
    if not pool_mode:
        pool_mode = "attn" if (isinstance(ck, dict) and "pool_key" in ck) else "mean"
    del ck

    comp = SoftTokenCompressor(
        decoder,
        factor=args.factor,
        enc_layers=args.enc_layers,
        train_encoder=train_encoder,
        mode="simple",
        pool_mode=pool_mode,
    ).cuda()
    comp.load_trained(args.softtoken_ckpt)
    comp.eval()
    print(f"[lb-soft] loaded {args.softtoken_ckpt} "
          f"(factor={args.factor}, pool={pool_mode}, train_encoder={train_encoder})")
    return comp


def split_prompt(template, row):
    marker = "{context}"
    if marker not in template:
        raise ValueError("LongBench prompt template must contain {context}")
    left, right = template.split(marker, 1)
    left = left.format(context="", input=row.get("input", ""))
    right = right.format(context="", input=row.get("input", ""))
    return left, right


def generate_text(decoder, tok, prompt, max_new_tokens):
    ids = tok(prompt, truncation=False, return_tensors="pt").input_ids.to(decoder.device)
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


def generate_soft(decoder, tok, comp, left, context, right, max_new_tokens, enc_window):
    ctx_ids = tok(context, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    mask = torch.ones_like(ctx_ids)
    with torch.no_grad():
        if enc_window and enc_window > 0:
            soft = comp.encode_long(ctx_ids, mask, window=enc_window)
        else:
            soft = comp.encode(ctx_ids, mask)
        left_ids = tok(left, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        right_ids = tok(right, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        left_emb = decoder.model.embed_tokens(left_ids).to(decoder.dtype)
        right_emb = decoder.model.embed_tokens(right_ids).to(decoder.dtype)
        prefix_len = left_emb.shape[1] + soft.shape[1] + right_emb.shape[1]
        inputs_embeds = torch.cat([
            left_emb,
            soft.to(decoder.device, dtype=decoder.dtype),
            right_emb,
        ], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=decoder.device)
        out = decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )[0]
    # With inputs_embeds, HF generation usually returns only generated token ids;
    # some versions include dummy prefix ids. Support both shapes.
    gen_ids = out[prefix_len:] if out.shape[0] > max_new_tokens + 2 else out
    pred = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return pred, int(ctx_ids.shape[1]), int(soft.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(QA_DATASETS))
    ap.add_argument("--data_dir", default=str(LONGBENCH_ROOT / "data"))
    ap.add_argument("--official_dir", default=str(LONGBENCH_ROOT / "official_eval"))
    ap.add_argument("--condition", choices=["raw", "softtoken"], default="softtoken")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--decoder", default="qwen3-8b")
    ap.add_argument("--softtoken_ckpt", default=None)
    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--enc_window", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_root", default=None,
                    help="Defaults to <official_dir>/pred/<run_name>.")
    args = ap.parse_args()

    official_dir = Path(args.official_dir)
    prompts = load_json(official_dir / "config" / "dataset2prompt.json")
    maxlens = load_json(official_dir / "config" / "dataset2maxlen.json")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = rv.resolve_model(args.decoder)
    print(f"[lb-soft] loading decoder {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    decoder = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    comp = load_softtoken(args, decoder) if args.condition == "softtoken" else None

    out_root = Path(args.out_root) if args.out_root else official_dir / "pred" / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {"condition": args.condition, "decoder": args.decoder,
            "softtoken_ckpt": args.softtoken_ckpt, "factor": args.factor,
            "datasets": [], "records": {}}

    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        rows = list(iter_jsonl(Path(args.data_dir) / f"{dataset}.jsonl"))
        if args.limit:
            rows = rows[:args.limit]
        out_path = out_root / f"{dataset}.jsonl"
        ratios = []
        print(f"[lb-soft] {dataset}: {len(rows)} examples -> {out_path}")
        with out_path.open("w", encoding="utf-8") as f:
            for row in tqdm(rows):
                template = prompts[dataset]
                max_new = int(maxlens[dataset])
                if args.condition == "raw":
                    prompt = template.format(context=row["context"], input=row["input"])
                    pred = generate_text(decoder, tok, prompt, max_new)
                else:
                    left, right = split_prompt(template, row)
                    pred, raw_tokens, soft_tokens = generate_soft(
                        decoder, tok, comp, left, row["context"], right,
                        max_new, args.enc_window)
                    ratios.append(raw_tokens / max(1, soft_tokens))
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
            "mean_compression_ratio": sum(ratios) / len(ratios) if ratios else 1.0,
        }
    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[lb-soft] wrote {out_root}")


if __name__ == "__main__":
    main()
