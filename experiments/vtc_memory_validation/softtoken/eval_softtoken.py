#!/usr/bin/env python3
"""
Evaluate a trained soft-token compressor on memory QA.

Unlike raw/summary/dsocr (which feed text token IDs), soft-token answering must
inject continuous vectors via `inputs_embeds`, so it gets its own eval script
(same pattern as the DeepSeek-OCR two-stage flow).

For each QA item:
  1. compress the conversation history to soft tokens (simple or full mode),
  2. generate an answer from [soft_tokens ; question_embeddings],
  3. judge correctness (same rules as run_validation), report per-category
     accuracy + achieved compression ratio.

Run on the pod (main env):
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python softtoken/eval_softtoken.py --config configs/simple.json \
      --ckpt softtoken/ckpt_simple_f8.pt --limit 20
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_validation as rv
from softtoken.compressor import SoftTokenCompressor, build_role_segments


QA_INSTR = ("\n\nAnswer the question using only the conversation above. "
            "If not present, reply: NOT MENTIONED.\nQuestion: ")


def generate_from_soft(comp, tok, soft, question, max_new_tokens=64):
    """soft: (1, M, d). Feed [soft ; question_emb] and greedily decode."""
    q_ids = tok(QA_INSTR + question + "\nAnswer:", return_tensors="pt")[
        "input_ids"].to(soft.device)
    q_emb = comp.embed_tokens(q_ids)
    inp = torch.cat([soft, q_emb], dim=1)
    attn = torch.ones(inp.shape[:2], dtype=torch.long, device=inp.device)
    with torch.no_grad():
        out = comp.decoder.generate(
            inputs_embeds=inp, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def judge_simple(tok, comp, question, gold, pred):
    ng, np_ = rv.normalize(gold), rv.normalize(pred)
    if ng and (ng == np_ or ng in np_ or np_ in ng):
        return True
    # lightweight judge using the same decoder over text tokens
    msg = (f"Question: {question}\nGold: {gold}\nAnswer: {pred}\n"
           f"Is the answer correct? Reply CORRECT or WRONG.\n")
    ids = tok(msg, return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        out = comp.decoder.generate(input_ids=ids, max_new_tokens=4, do_sample=False)
    verdict = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return "CORRECT" in verdict.upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/simple.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--factor", type=int, default=None,
                    help="override the config's compression factor (simple mode)")
    ap.add_argument("--decoder", default=None,
                    help="override the config's decoder/reader (registry name or path)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle before --limit (match run_validation for a "
                         "fair same-sample comparison on LongMemEval)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    dataset = args.dataset or cfg.get("dataset", "locomo")
    data = args.data or cfg.get("data", "data/locomo10.json")
    mode = cfg.get("mode", "simple")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = rv.resolve_model(args.decoder or cfg.get("decoder", "qwen2.5-7b"))
    print(f"[eval] loading decoder {path}")
    tok = AutoTokenizer.from_pretrained(path)
    decoder = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    comp = SoftTokenCompressor(
        decoder, factor=args.factor or cfg.get("factor", 8),
        enc_layers=cfg.get("enc_layers", 2), train_encoder=True, mode=mode,
        role_factors={"user": cfg.get("user_factor", 4),
                      "assistant": cfg.get("assistant_factor", 16)}).cuda()
    comp.load_trained(args.ckpt)
    comp.eval()
    print(f"[eval] loaded ckpt {args.ckpt} (mode={mode})")

    raw = rv.load_json(data)
    items = list(rv.iter_items(dataset, raw, limit_per_sample=None))
    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(items)
    items = items[: args.limit]
    print(f"[eval] {len(items)} QA items")

    results = defaultdict(list)
    ratios = []
    records = []
    for idx, (si, conv_text, question, gold, cat) in enumerate(items):
        full_ids = tok(conv_text, return_tensors="pt")["input_ids"].cuda()
        n_tok = full_ids.shape[1]

        # build soft tokens
        with torch.no_grad():
            if mode == "full":
                turns = _turns_for_sample(dataset, raw, si)
                ids, segs = build_role_segments(tok, turns)
                ids = ids.cuda()
                soft_list = comp.encode(ids, torch.ones_like(ids), segments=segs)
                soft = soft_list[0].unsqueeze(0)
                n_soft = soft.shape[1]
            else:
                mask = torch.ones_like(full_ids)
                soft = comp.encode(full_ids, mask)  # (1, M, d)
                n_soft = soft.shape[1]

        pred = generate_from_soft(comp, tok, soft.to(decoder.dtype), question)
        ok = judge_simple(tok, comp, question, gold, pred)
        results[cat].append(ok)
        ratios.append(n_tok / max(1, n_soft))
        records.append({"i": idx, "category": cat, "question": question,
                        "gold": gold, "pred": pred, "ok": ok,
                        "tokens": n_tok, "soft_tokens": n_soft})
        if (idx + 1) % 5 == 0:
            print(f"  ... {idx + 1}/{len(items)}")

    print("\n============ SOFTTOKEN RESULTS ============")
    for cat in sorted(results):
        b = results[cat]
        print(f"{cat:<16} {sum(b)/len(b):.3f}  (n={len(b)})")
    allb = [x for v in results.values() for x in v]
    print(f"{'OVERALL':<16} {sum(allb)/len(allb):.3f}")
    print(f"mean compression: {sum(ratios)/len(ratios):.2f}x")

    out = args.out or f"results_softtoken_{cfg.get('name','x')}.json"
    with open(out, "w") as f:
        json.dump({"config": cfg, "ckpt": args.ckpt, "records": records}, f, indent=2)
    print(f"[eval] wrote {out}")


def _turns_for_sample(dataset_name, raw, si):
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


if __name__ == "__main__":
    main()
