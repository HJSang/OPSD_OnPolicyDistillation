#!/usr/bin/env python3
"""
Minimal training loop for the soft-token compressor prototype.

Objective (per step, decoder frozen, only adapter/gate trained):
  - reconstruction: feed soft tokens, teacher-force the ORIGINAL text back,
    minimize cross-entropy (autoencoder — can the k soft tokens carry the text?).
  - forward-KL (optional): match the frozen decoder's next-token distribution
    from the compressed context to the full-text context.

This prototype trains on short text chunks (single block, no session chunking)
just to prove "compress -> inject -> reconstruct" works end to end and the loss
goes down. Scale up later.

Run on the pod (main env, transformers 5.x):
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python softtoken/train.py --decoder qwen2.5-7b --factor 8 --steps 200 \
      --data data/locomo10.json --dataset locomo --smoke
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_validation as rv  # resolve_model, data loaders
from softtoken.compressor import SoftTokenCompressor


def make_chunks(dataset_name, data_path, tokenizer, max_len, n_chunks):
    """Turn conversations into token chunks for autoencoding.

    Conversations shorter than max_len are padded to max_len (previously they
    were silently dropped, which starved training when max_len exceeded the
    typical conversation length). Longer conversations are split into max_len
    windows. Returns (chunk_ids, chunk_mask) tensors so the training loop can
    ignore pad positions.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    data = rv.load_json(data_path)
    items = list(rv.iter_items(dataset_name, data, limit_per_sample=1))
    chunks, masks = [], []

    def _add(ids):
        L = len(ids)
        if L < max_len:
            pad = torch.full((max_len - L,), pad_id, dtype=ids.dtype)
            m = torch.cat([torch.ones(L, dtype=torch.long),
                           torch.zeros(max_len - L, dtype=torch.long)])
            chunks.append(torch.cat([ids, pad]))
            masks.append(m)
        else:
            chunks.append(ids[:max_len])
            masks.append(torch.ones(max_len, dtype=torch.long))

    for _, conv_text, *_ in items:
        ids = tokenizer(conv_text, return_tensors="pt")["input_ids"][0]
        if len(ids) <= max_len:
            _add(ids)
            if len(chunks) >= n_chunks:
                return chunks, masks
        else:
            for i in range(0, len(ids) - max_len + 1, max_len):
                _add(ids[i:i + max_len])
                if len(chunks) >= n_chunks:
                    return chunks, masks
    return chunks, masks


def _first_conversation_turns(dataset_name, raw):
    """Return the first conversation as a flat list of {role, content} turns."""
    if dataset_name == "ultrachat":
        return [{"role": t.get("role", "user"), "content": t.get("content", "")}
                for t in raw[0].get("turns", [])]
    if dataset_name == "longmemeval":
        inst = raw[0]
        turns = []
        for session in inst.get("haystack_sessions", []):
            for t in session:
                turns.append({"role": t.get("role", "user"),
                              "content": t.get("content", "")})
        return turns
    # locomo
    conv = raw[0]["conversation"]
    import re
    sids = sorted(int(k.split("_")[1]) for k in conv
                  if re.fullmatch(r"session_\d+", k))
    turns = []
    for sid in sids:
        for t in conv[f"session_{sid}"]:
            speaker = t.get("speaker", "")
            role = "user" if speaker == conv.get("speaker_a") else "assistant"
            turns.append({"role": role, "content": t.get("text", "")})
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default="qwen2.5-7b")
    ap.add_argument("--dataset", default="locomo",
                    choices=["locomo", "longmemeval", "ultrachat"])
    ap.add_argument("--data", default="data/locomo10.json")
    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--mode", default="simple", choices=["simple", "full"],
                    help="simple=uniform pooling; full=per-turn pooling with "
                         "per-role factors (our method)")
    ap.add_argument("--user_factor", type=int, default=4)
    ap.add_argument("--assistant_factor", type=int, default=16)
    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--train_encoder", action="store_true")
    ap.add_argument("--max_len", type=int, default=256, help="tokens per chunk")
    ap.add_argument("--n_chunks", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fkl_weight", type=float, default=0.0,
                    help="weight on forward-KL to full-text next-token dist")
    ap.add_argument("--config", default=None,
                    help="JSON config (configs/simple.json or configs/full.json) "
                         "whose keys set defaults for the args below; explicit "
                         "CLI flags still override.")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify")
    ap.add_argument("--save", default="softtoken/ckpt.pt")
    args = ap.parse_args()

    # apply config file (skip _comment / name); CLI-provided flags win
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        provided = {a.split("=")[0].lstrip("-").replace("-", "_")
                    for a in sys.argv[1:] if a.startswith("--")}
        for k, v in cfg.items():
            if k in ("_comment", "name"):
                continue
            if hasattr(args, k) and k not in provided:
                setattr(args, k, v)
        print(f"[st] loaded config {args.config} "
              f"(name={cfg.get('name', '?')}, mode={args.mode})")

    if args.smoke:
        args.steps, args.n_chunks, args.max_len = 20, 8, 128

    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = rv.resolve_model(args.decoder)
    print(f"[st] loading decoder {path}")
    tok = AutoTokenizer.from_pretrained(path)
    decoder = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda")
    decoder.eval()

    comp = SoftTokenCompressor(
        decoder, factor=args.factor, enc_layers=args.enc_layers,
        train_encoder=args.train_encoder, mode=args.mode,
        role_factors={"user": args.user_factor,
                      "assistant": args.assistant_factor}).cuda()

    trainable = [p for p in comp.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[st] trainable params: {n_train/1e6:.2f}M "
          f"(factor={args.factor}, enc_layers={args.enc_layers})")
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    print(f"[st] building {args.n_chunks} chunks of {args.max_len} tokens")
    chunks, chunk_masks = make_chunks(args.dataset, args.data, tok, args.max_len, args.n_chunks)
    print(f"[st] got {len(chunks)} chunks")

    # full-mode self-test: exercise per-turn role-based pooling on one real
    # conversation and report how many soft tokens each role produces.
    if args.mode == "full":
        from softtoken.compressor import build_role_segments
        raw = rv.load_json(args.data)
        turns = _first_conversation_turns(args.dataset, raw)
        ids, segs = build_role_segments(tok, turns, max_len=args.max_len)
        ids = ids.cuda()
        soft_list = comp.encode(ids, torch.ones_like(ids), segments=segs)
        n_soft = soft_list[0].shape[0]
        n_tok = ids.shape[1]
        print(f"[st][full] {len(segs[0])} turns, {n_tok} tokens -> {n_soft} soft "
              f"tokens ({n_tok/max(1,n_soft):.1f}x) with "
              f"user_factor={args.user_factor} assistant_factor={args.assistant_factor}")
        _ = comp.forward_with_soft_list(soft_list, ids[:, :args.max_len // 2])
        print("[st][full] per-turn encode + decode OK")

    import random
    rng = random.Random(0)
    comp.train()
    n_ch = len(chunks)
    for step in range(args.steps):
        idxs = [rng.randrange(n_ch) for _ in range(args.batch_size)]
        batch = torch.stack([chunks[i] for i in idxs]).cuda()
        mask = torch.stack([chunk_masks[i] for i in idxs]).cuda()

        soft = comp.encode(batch, mask)                     # (B, M, d)
        logits = comp.forward_with_soft(soft, batch, mask)  # reconstruct original
        # next-token CE: predict token t from positions <t (teacher forcing).
        # Ignore padding positions (mask==0) in the target.
        tgt = batch[:, 1:].clone()
        tgt[mask[:, 1:] == 0] = -100
        recon = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            tgt.reshape(-1), ignore_index=-100)

        loss = recon
        fkl = torch.tensor(0.0)
        if args.fkl_weight > 0:
            with torch.no_grad():
                full = decoder(input_ids=batch, attention_mask=mask).logits
            p_full = F.log_softmax(full[:, :-1].float(), -1)
            p_comp = F.log_softmax(logits[:, :-1].float(), -1)
            # Per-position KL, then average over non-padding positions only.
            # (batchmean divided by batch size, not token count, which made fkl
            # scale with sequence length and blow up once padding was added.)
            kl_per_pos = F.kl_div(
                p_comp, p_full, log_target=True, reduction="none").sum(-1)
            valid = mask[:, 1:].float()
            fkl = (kl_per_pos * valid).sum() / valid.sum().clamp(min=1.0)
            loss = recon + args.fkl_weight * fkl

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % 5 == 0 or step == args.steps - 1:
            ppl = torch.exp(recon.detach()).item()
            print(f"  step {step:4d} | recon {recon.item():.3f} "
                  f"(ppl {ppl:.1f}) | fkl {float(fkl):.3f}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    ckpt = {"adapter": comp.adapter.state_dict(),
            "gate": comp.gate.state_dict(),
            "args": vars(args)}
    # if we trained the borrowed encoder layers (deep-copied), persist them too
    if args.train_encoder:
        ckpt["enc_layers"] = comp.layers.state_dict()
    torch.save(ckpt, args.save)
    print(f"[st] saved trainable head"
          f"{' + encoder layers' if args.train_encoder else ''} to {args.save}")


if __name__ == "__main__":
    main()
