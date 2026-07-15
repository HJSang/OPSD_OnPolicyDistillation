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

Example:
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


def _iter_text_items(dataset_name, data):
    if dataset_name == "synthlocomo":
        for i, item in enumerate(data):
            passage = item.get("passage", "")
            if passage:
                yield i, passage, "", "", "synthlocomo"
        return
    if dataset_name == "msc_lme":
        seen = set()
        for i, item in enumerate(data):
            key = item.get("source_conversation_index", i)
            if key in seen:
                continue
            seen.add(key)
            passage = rv.msc_lme_to_text(item)
            if passage:
                yield i, passage, "", "", "msc_lme"
        return
    yield from rv.iter_items(dataset_name, data, limit_per_sample=1)


def make_long_examples(dataset_name, data_path, tokenizer, max_mem_tokens,
                       target_len, n_examples, min_mem_tokens=512):
    """Build (context_ids, target_ids) pairs for long-context training.

    Each example compresses a long context span (up to `max_mem_tokens`, so the
    resulting soft-token memory reaches the length the decoder must read at eval
    on long conversations) and is trained to predict the `target_len` tokens
    that immediately follow it. This aligns the training soft-memory length with
    evaluation, fixing the train/eval length mismatch that made whole-
    conversation compression degenerate on long benchmarks (e.g. LoCoMo).

    Training corpora such as UltraChat have short conversations (median ~1k
    tokens), far below the eval length (LoCoMo median ~22k). We therefore
    CONCATENATE consecutive conversations into a synthetic long dialogue until it
    reaches the sampled memory length, then predict the next `target_len` tokens.
    This exposes the decoder to eval-scale soft memories using only short-
    conversation data, without touching any evaluation benchmark.

    The context length is sampled per example in [min_mem_tokens, max_mem_tokens]
    so the decoder sees a range of memory lengths (curriculum), not one fixed
    length. Returns a list of (context_ids[1,N], target_ids[1,T]) tensor pairs.
    """
    import random
    rng = random.Random(0)
    data = rv.load_json(data_path)
    items = list(_iter_text_items(dataset_name, data))
    # pre-tokenize every conversation once
    conv_ids = []
    for _, conv_text, *_ in items:
        ids = tokenizer(conv_text, return_tensors="pt")["input_ids"][0]
        if len(ids) > 8:
            conv_ids.append(ids)
    if not conv_ids:
        return []

    examples = []
    n_conv = len(conv_ids)
    guard = 0
    while len(examples) < n_examples and guard < n_examples * 50:
        guard += 1
        mem = rng.randint(min_mem_tokens, max_mem_tokens)
        need = mem + target_len
        # concatenate consecutive conversations (random start) until long enough
        buf = []
        total = 0
        j = rng.randrange(n_conv)
        while total < need and len(buf) < 200:
            c = conv_ids[j % n_conv]
            buf.append(c)
            total += len(c)
            j += 1
        if total < need:
            continue
        cat = torch.cat(buf)[:need]
        ctx = cat[:mem]
        tgt = cat[mem:mem + target_len]
        examples.append((ctx.unsqueeze(0), tgt.unsqueeze(0)))
    return examples


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
    items = list(_iter_text_items(dataset_name, data))
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
        turns = raw[0].get("turns") or raw[0].get("messages") or []
        return [{"role": t.get("role", "user"), "content": t.get("content", "")}
                for t in turns]
    if dataset_name == "longmemeval":
        inst = raw[0]
        turns = []
        for session in inst.get("haystack_sessions", []):
            for t in session:
                turns.append({"role": t.get("role", "user"),
                              "content": t.get("content", "")})
        return turns
    if dataset_name == "msc_lme":
        return [{"role": t.get("role", "user"), "content": t.get("content", "")}
                for s in raw[0].get("haystack_sessions", []) for t in s]
    if dataset_name == "synthlocomo":
        turns = []
        for session in raw[0].get("sessions", []):
            for t in session.get("turns", []):
                turns.append({"role": t.get("speaker", "A"),
                              "content": t.get("text", "")})
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


def _save_ckpt(args, comp):
    save_dir = os.path.dirname(os.path.abspath(args.save))
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {"adapter": comp.adapter.state_dict(),
            "gate": comp.gate.state_dict(),
            "args": vars(args)}
    if args.train_encoder:
        ckpt["enc_layers"] = comp.layers.state_dict()
    if getattr(comp, "pool_heads", 0) > 1:
        ckpt["pool_keys"] = comp.pool_keys.state_dict()
        ckpt["pool_query"] = comp.pool_query.detach().cpu()
    elif getattr(comp, "pool_heads", 0) == 1:
        ckpt["pool_key"] = comp.pool_key.state_dict()
        ckpt["pool_query"] = comp.pool_query.detach().cpu()
    torch.save(ckpt, args.save)
    print(f"[st] saved trainable head"
          f"{' + encoder layers' if args.train_encoder else ''} to {args.save}")


def _run_long_context_training(args, comp, decoder, tok, opt):
    """Train the decoder to read long soft-token memories.

    Compress a long context span into a long soft memory (via windowed encoding
    so the encoder stays in-distribution), then teacher-force the held-out
    continuation. This matches the eval regime (whole-conversation compression
    -> long soft memory) that fixed-chunk reconstruction never exercised, so the
    decoder learns to attend over soft tokens at eval-scale positions instead of
    degenerating on LoCoMo-length inputs.
    """
    import random
    rng = random.Random(0)
    print(f"[st][long] building long-context examples up to "
          f"{args.max_mem_tokens} ctx tokens (target {args.target_len})")
    examples = make_long_examples(
        args.dataset, args.data, tok, args.max_mem_tokens, args.target_len,
        n_examples=args.n_chunks, min_mem_tokens=args.min_mem_tokens)
    print(f"[st][long] got {len(examples)} examples")
    if not examples:
        raise RuntimeError("no long-context examples built; check data/lengths")

    # Gradient checkpointing on the frozen decoder with inputs_embeds + reentrant
    # autograd produces NaN grads in bf16, so it is disabled by default. The
    # trainable params (adapter/gate/enc_layers) are small and only the target
    # window (short) needs decoder activations, so memory stays bounded even at
    # eval-scale soft memories. Enable only if you hit OOM at very long memories.
    if args.grad_checkpoint and hasattr(decoder, "gradient_checkpointing_enable"):
        decoder.gradient_checkpointing_enable()

    comp.train()
    n = len(examples)
    for step in range(args.steps):
        ctx, tgt = examples[rng.randrange(n)]
        ctx = ctx.cuda(); tgt = tgt.cuda()
        soft = comp.encode_long(ctx, window=args.enc_window)      # (1, M, d)
        logits = comp.forward_with_soft(soft, tgt)                # (1, T, V)
        # next-token CE on the held-out continuation
        pred = logits[:, :-1].reshape(-1, logits.size(-1)).float()
        gold = tgt[:, 1:].reshape(-1)
        recon = F.cross_entropy(pred, gold)

        loss = recon
        fkl = torch.tensor(0.0)
        if args.fkl_weight > 0:
            with torch.no_grad():
                full = decoder(input_ids=torch.cat([ctx, tgt], dim=1)).logits
                full = full[:, ctx.shape[1]:, :]                 # target positions
            p_full = F.log_softmax(full[:, :-1].float(), -1)
            p_comp = F.log_softmax(logits[:, :-1].float(), -1)
            # per-token KL (mean over target positions), matching the short-chunk
            # loop's normalization. reduction="batchmean" divides only by batch
            # size, so on a length-T target it summed T positions and exploded
            # (fkl~125, grad-norm~1000) once the target window was long.
            kl_per_pos = F.kl_div(p_comp, p_full, log_target=True,
                                  reduction="none").sum(-1)
            fkl = kl_per_pos.mean()
            loss = recon + args.fkl_weight * fkl

        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in comp.parameters() if p.requires_grad], 1.0)
        # skip steps that produced non-finite loss/grads (bf16 long-context can
        # occasionally overflow); this keeps a bad batch from poisoning weights.
        if not torch.isfinite(loss) or not torch.isfinite(gnorm):
            opt.zero_grad(set_to_none=True)
            if step % 5 == 0:
                print(f"  [long] step {step:4d} | SKIP non-finite "
                      f"(loss={float(loss)}, gnorm={float(gnorm)})")
            continue
        opt.step()

        if step % 5 == 0 or step == args.steps - 1:
            ppl = torch.exp(recon.detach()).item()
            print(f"  [long] step {step:4d} | M={soft.shape[1]:4d} "
                  f"| recon {recon.item():.3f} (ppl {ppl:.1f}) "
                  f"| fkl {float(fkl):.3f} | gnorm {float(gnorm):.2f}")


QA_INSTR = ("\n\nAnswer the question using only the conversation above. "
            "If not present, reply: NOT MENTIONED.\nQuestion: ")
QA_INSTR_NO_ABSTAIN = ("\n\nAnswer the question using only the conversation above. "
                       "Give a short answer.\nQuestion: ")


def _qa_instr(args):
    return QA_INSTR_NO_ABSTAIN if getattr(args, "no_abstain", False) else QA_INSTR


def _load_qa_items(path):
    """Load passage-QA or LME-style rows and flatten to (passage, q, a) triples."""
    data = rv.load_json(path)
    items = []
    for ex in data:
        if "haystack_sessions" in ex and "question" in ex and "answer" in ex:
            passage = rv.longmemeval_to_text(ex)
            q, a = ex.get("question", ""), ex.get("answer", "")
            if passage and q and a:
                items.append((passage, q, str(a)))
            continue
        passage = ex.get("passage", "")
        for qa in ex.get("qas", []):
            q, a = qa.get("q", ""), qa.get("a", "")
            if passage and q and a:
                items.append((passage, q, a))
    return items


def _answer_ids(tok, answer):
    ids = tok(" " + answer, return_tensors="pt",
              add_special_tokens=False)["input_ids"]
    if tok.eos_token_id is not None:
        eos = torch.tensor([[tok.eos_token_id]], dtype=ids.dtype)
        ids = torch.cat([ids, eos], dim=1)
    return ids.cuda()


def _qa_answer_loss(comp, decoder, tok, passage, question, answer, max_len,
                    qa_instr=QA_INSTR):
    """Compress `passage`, feed [soft; question; answer], return CE on answer
    tokens only. Shared by pure-QA and QA+reconstruction training."""
    p_ids = tok(passage, return_tensors="pt",
                truncation=True, max_length=max_len)["input_ids"].cuda()
    q_ids = tok(qa_instr + question + "\nAnswer:",
                return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    a_ids = _answer_ids(tok, answer)
    soft = comp.encode(p_ids, torch.ones_like(p_ids))
    if isinstance(soft, list):
        soft = soft[0].unsqueeze(0)
    q_emb = comp.embed_tokens(q_ids).to(soft.dtype)
    a_emb = comp.embed_tokens(a_ids).to(soft.dtype)
    inp = torch.cat([soft, q_emb, a_emb], dim=1)
    attn = torch.ones(inp.shape[:2], dtype=torch.long, device=inp.device)
    logits = decoder(inputs_embeds=inp, attention_mask=attn).logits
    A = a_ids.shape[1]
    ans_logits = logits[:, -A - 1:-1, :]
    return F.cross_entropy(
        ans_logits.reshape(-1, ans_logits.size(-1)).float(),
        a_ids.reshape(-1)), soft.shape[1]


def _recon_loss(comp, decoder, tok, chunk, mask, fkl_weight):
    """Reconstruction CE (+optional fkl) for one chunk. Shared helper."""
    batch = chunk.unsqueeze(0).cuda()
    m = mask.unsqueeze(0).cuda()
    soft = comp.encode(batch, m)
    logits = comp.forward_with_soft(soft, batch, m)
    tgt = batch[:, 1:].clone()
    tgt[m[:, 1:] == 0] = -100
    recon = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        tgt.reshape(-1), ignore_index=-100)
    if fkl_weight > 0:
        with torch.no_grad():
            full = decoder(input_ids=batch, attention_mask=m).logits
        p_full = F.log_softmax(full[:, :-1].float(), -1)
        p_comp = F.log_softmax(logits[:, :-1].float(), -1)
        kl = F.kl_div(p_comp, p_full, log_target=True,
                      reduction="none").sum(-1)
        valid = m[:, 1:].float()
        recon = recon + fkl_weight * (kl * valid).sum() / valid.sum().clamp(min=1.0)
    return recon


def _run_qa_recon_training(args, comp, decoder, tok, opt):
    """Scheme B: joint reconstruction + QA training.

    Each micro-step adds a RECONSTRUCTION loss on an UltraChat chunk (keeps the
    compressor's general information-preservation ability -- the recipe that
    scores 0.207 on LoCoMo) and a QA loss on a RepLiQA (passage, q, a) triple
    (adds an information-extraction signal that highlights answerable facts).
    loss = recon + qa_weight * qa. Reconstruction data (--recon_data, dialogue)
    and QA data (--data, passage-QA) are kept SEPARATE so reconstruction stays
    in the conversational domain that matches LoCoMo while QA teaches extraction.
    """
    import random
    rng = random.Random(0)
    qa_items = _load_qa_items(args.data)
    qa_instr = _qa_instr(args)
    print(f"[st][qa+recon] loaded {len(qa_items)} QA triples from {args.data}")
    chunks, chunk_masks = make_chunks(
        "ultrachat", args.recon_data, tok, args.max_len, args.n_chunks)
    print(f"[st][qa+recon] loaded {len(chunks)} recon chunks from {args.recon_data}")
    if not qa_items or not chunks:
        raise RuntimeError("need both QA triples and reconstruction chunks")
    comp.train()
    nqa, nch = len(qa_items), len(chunks)
    accum = max(1, args.qa_accum)
    opt.zero_grad()
    run_qa = run_rc = 0.0
    for step in range(args.steps):
        qa_sum = rc_sum = 0.0
        for _ in range(accum):
            # reconstruction on a dialogue chunk
            i = rng.randrange(nch)
            rc = _recon_loss(comp, decoder, tok, chunks[i], chunk_masks[i],
                             args.fkl_weight) / accum
            # QA on a passage-QA triple
            p, q, a = qa_items[rng.randrange(nqa)]
            qa, M = _qa_answer_loss(
                comp, decoder, tok, p, q, a, args.max_len, qa_instr)
            qa = qa / accum
            loss = rc + args.qa_weight * qa
            if torch.isfinite(loss):
                loss.backward()
                rc_sum += float(rc) * accum
                qa_sum += float(qa) * accum
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in comp.parameters() if p.requires_grad], 1.0)
        if torch.isfinite(gnorm):
            opt.step()
        opt.zero_grad(set_to_none=True)
        a = 0.1 if step else 1.0
        run_qa = (1 - a) * run_qa + a * (qa_sum / accum)
        run_rc = (1 - a) * run_rc + a * (rc_sum / accum)
        if step % 5 == 0 or step == args.steps - 1:
            print(f"  [qa+recon] step {step:4d} "
                  f"| recon {run_rc:.3f} | qa {run_qa:.3f} "
                  f"| gnorm {float(gnorm):.2f}")


def _manifest_paths(manifest_path, entry):
    base = os.path.dirname(os.path.abspath(manifest_path))
    raw = []
    if entry.get("path"):
        raw.append(entry["path"])
    raw.extend(entry.get("paths") or [])
    paths = []
    for p in raw:
        if not p:
            continue
        if os.path.isabs(p):
            paths.append(p)
            continue
        cwd_path = os.path.abspath(p)
        manifest_path = os.path.normpath(os.path.join(base, p))
        # Prefer paths that are valid from the training working directory, but
        # also support manifests with entries relative to the manifest file.
        # This keeps old manifests working and prevents memory_mix/data-relative
        # entries from accidentally resolving to memory_mix/data/memory_mix/data.
        if os.path.exists(cwd_path):
            paths.append(cwd_path)
        elif os.path.exists(manifest_path):
            paths.append(manifest_path)
        else:
            paths.append(cwd_path)
    return paths


def _load_mix_streams(args, tok):
    manifest = rv.load_json(args.mix_manifest)
    mix = manifest.get("mix", manifest)
    streams = []
    for name, entry in mix.items():
        if not entry.get("enabled", True):
            continue
        weight = float(entry.get("weight", 0.0))
        if weight <= 0:
            continue
        kind = entry.get("kind")
        paths = _manifest_paths(args.mix_manifest, entry)
        if not paths:
            continue
        if kind == "recon":
            chunks, masks = [], []
            for p in paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"mix_manifest stream {name} missing recon path: {p}")
                c, m = make_chunks("ultrachat", p, tok, args.max_len, args.n_chunks)
                chunks.extend(c)
                masks.extend(m)
            if chunks:
                streams.append({
                    "name": name,
                    "kind": "recon",
                    "weight": weight,
                    "chunks": chunks,
                    "masks": masks,
                })
                print(f"[st][mix] stream {name}: recon chunks={len(chunks)} weight={weight:.4f}")
        elif kind == "qa":
            items = []
            for p in paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"mix_manifest stream {name} missing QA path: {p}")
                items.extend(_load_qa_items(p))
            if items:
                streams.append({
                    "name": name,
                    "kind": "qa",
                    "weight": weight,
                    "items": items,
                })
                print(f"[st][mix] stream {name}: QA triples={len(items)} weight={weight:.4f}")
        else:
            raise ValueError(f"mix_manifest stream {name} has unknown kind={kind!r}")
    total = sum(s["weight"] for s in streams)
    if not streams or total <= 0:
        raise RuntimeError("mix_manifest produced no enabled non-empty streams")
    for s in streams:
        s["weight"] /= total
    print("[st][mix] normalized weights: " +
          ", ".join(f"{s['name']}={s['weight']:.3f}" for s in streams))
    return streams


def _run_mix_manifest_training(args, comp, decoder, tok, opt):
    """Train from a manifest of recon and QA streams.

    Each micro-step samples exactly one stream according to manifest weights.
    This makes the target mix explicit and avoids the old fixed pattern of one
    reconstruction example plus one QA example every step.
    """
    import random
    rng = random.Random(0)
    streams = _load_mix_streams(args, tok)
    qa_instr = _qa_instr(args)
    weights = [s["weight"] for s in streams]
    accum = max(1, args.qa_accum)
    comp.train()
    opt.zero_grad()
    run_qa = run_rc = 0.0
    for step in range(args.steps):
        qa_sum = rc_sum = 0.0
        qa_n = rc_n = 0
        picked = {s["name"]: 0 for s in streams}
        for _ in range(accum):
            s = rng.choices(streams, weights=weights, k=1)[0]
            picked[s["name"]] += 1
            if s["kind"] == "recon":
                i = rng.randrange(len(s["chunks"]))
                loss = _recon_loss(comp, decoder, tok, s["chunks"][i], s["masks"][i],
                                   args.fkl_weight)
                rc_sum += float(loss.detach())
                rc_n += 1
            else:
                p, q, a = s["items"][rng.randrange(len(s["items"]))]
                loss, _ = _qa_answer_loss(
                    comp, decoder, tok, p, q, a, args.max_len, qa_instr)
                loss = args.qa_weight * loss
                qa_sum += float(loss.detach()) / max(args.qa_weight, 1e-12)
                qa_n += 1
            loss = loss / accum
            if torch.isfinite(loss):
                loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in comp.parameters() if p.requires_grad], 1.0)
        if torch.isfinite(gnorm):
            opt.step()
        opt.zero_grad(set_to_none=True)
        a = 0.1 if step else 1.0
        if qa_n:
            run_qa = (1 - a) * run_qa + a * (qa_sum / qa_n)
        if rc_n:
            run_rc = (1 - a) * run_rc + a * (rc_sum / rc_n)
        if step % 5 == 0 or step == args.steps - 1:
            mix_counts = ",".join(f"{k}:{v}" for k, v in picked.items() if v)
            print(f"  [mix] step {step:4d} | recon {run_rc:.3f} | qa {run_qa:.3f} "
                  f"| picks {mix_counts} | gnorm {float(gnorm):.2f}")


def _run_qa_training(args, comp, decoder, tok, opt):
    """QA-aware training: compress a passage into soft tokens, then feed
    [soft ; question ; answer] and minimize cross-entropy ONLY on the answer
    tokens. This trains the compressor to preserve the information needed to
    answer questions (information extraction) rather than to reconstruct the
    passage verbatim -- directly targeting the fine-grained-fact loss that
    reconstruction training misses. Matches the eval regime ([soft ; question]
    -> answer) far better than autoencoding.
    """
    import random
    rng = random.Random(0)
    items = _load_qa_items(args.data)
    qa_instr = _qa_instr(args)
    print(f"[st][qa] loaded {len(items)} (passage, q, a) triples")
    if not items:
        raise RuntimeError("no QA triples; check --data format")
    comp.train()
    n = len(items)
    accum = max(1, args.qa_accum)
    opt.zero_grad()
    running = 0.0
    for step in range(args.steps):
        # accumulate gradients over `accum` QA triples to reduce the high
        # variance of single-example answer CE before each optimizer step.
        loss_sum = 0.0
        for _ in range(accum):
            passage, question, answer = items[rng.randrange(n)]
            p_ids = tok(passage, return_tensors="pt",
                        truncation=True, max_length=args.max_len)["input_ids"].cuda()
            q_ids = tok(qa_instr + question + "\nAnswer:",
                        return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
            a_ids = _answer_ids(tok, answer)

            soft = comp.encode(p_ids, torch.ones_like(p_ids))       # (1, M, d)
            if isinstance(soft, list):
                soft = soft[0].unsqueeze(0)
            q_emb = comp.embed_tokens(q_ids).to(soft.dtype)
            a_emb = comp.embed_tokens(a_ids).to(soft.dtype)
            inp = torch.cat([soft, q_emb, a_emb], dim=1)            # (1, M+Q+A, d)
            attn = torch.ones(inp.shape[:2], dtype=torch.long, device=inp.device)
            out = decoder(inputs_embeds=inp, attention_mask=attn)
            logits = out.logits                                     # (1, L, V)

            # answer tokens occupy the last A positions; predict each answer
            # token from the position immediately before it (teacher forcing).
            A = a_ids.shape[1]
            ans_logits = logits[:, -A - 1:-1, :]                    # (1, A, V)
            loss = F.cross_entropy(
                ans_logits.reshape(-1, ans_logits.size(-1)).float(),
                a_ids.reshape(-1)) / accum
            if torch.isfinite(loss):
                loss.backward()
                loss_sum += float(loss) * accum

        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in comp.parameters() if p.requires_grad], 1.0)
        if torch.isfinite(gnorm):
            opt.step()
        opt.zero_grad(set_to_none=True)

        running = 0.9 * running + 0.1 * (loss_sum / accum) if step else loss_sum / accum
        if step % 5 == 0 or step == args.steps - 1:
            ppl = torch.exp(torch.tensor(running)).item()
            print(f"  [qa] step {step:4d} | M={soft.shape[1]:4d} "
                  f"| ans_ce(ema) {running:.3f} (ppl {ppl:.1f}) "
                  f"| gnorm {float(gnorm):.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default="qwen2.5-7b")
    ap.add_argument("--dataset", default="locomo",
                    choices=["locomo", "longmemeval", "ultrachat", "synthlocomo", "msc_lme"])
    ap.add_argument("--data", default=os.path.join(rv.DATA_DIR, "locomo10.json"))
    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--mode", default="simple", choices=["simple", "full"],
                    help="simple=uniform pooling; full=per-turn pooling with "
                         "per-role factors (our method)")
    ap.add_argument("--user_factor", type=int, default=4)
    ap.add_argument("--assistant_factor", type=int, default=16)
    ap.add_argument("--pool", default="mean", choices=["mean", "attn", "attn4"],
                    help="token pooling: mean, attn (single query attention), "
                         "or attn4 (four independent attention heads averaged; "
                         "about 5x trainable params vs mean for Qwen3-8B). All "
                         "attention pools initialize to exact mean pooling.")
    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--train_encoder", action="store_true")
    ap.add_argument("--max_len", type=int, default=256, help="tokens per chunk")
    ap.add_argument("--n_chunks", type=int, default=64)
    ap.add_argument("--long_context", action="store_true",
                    help="train the decoder to read long soft-token memories "
                         "(compress a long context, predict the next window). "
                         "Fixes the train/eval length mismatch that makes "
                         "whole-conversation compression degenerate on LoCoMo.")
    ap.add_argument("--qa_train", action="store_true",
                    help="QA-aware training: compress a passage, feed "
                         "[soft; question; answer], loss only on answer tokens. "
                         "Trains information extraction (data: [{passage,qas}]).")
    ap.add_argument("--no_abstain", action="store_true",
                    help="[qa_train/qa_recon/mix_manifest] train with an "
                         "answerable-only QA prompt instead of NOT MENTIONED.")
    ap.add_argument("--qa_accum", type=int, default=8,
                    help="[qa_train] gradient-accumulation micro-batch size to "
                         "reduce single-example answer-CE variance.")
    ap.add_argument("--qa_recon", action="store_true",
                    help="Scheme B: joint reconstruction (--recon_data dialogue) "
                         "+ QA (--data passage-QA). loss = recon + qa_weight*qa.")
    ap.add_argument("--mix_manifest", default=None,
                    help="Manifest with weighted recon/QA streams. When set, "
                         "training samples streams by manifest weights instead "
                         "of using only --data/--recon_data.")
    ap.add_argument("--recon_data", default=os.path.join(
        rv.DATA_DIR, "ultrachat_train.json"),
                    help="[qa_recon] dialogue corpus for the reconstruction term.")
    ap.add_argument("--qa_weight", type=float, default=1.0,
                    help="[qa_recon] weight on the QA term relative to recon.")
    ap.add_argument("--max_mem_tokens", type=int, default=4096,
                    help="[long_context] max context tokens per example; the "
                         "soft memory reaches ~max_mem_tokens/factor tokens.")
    ap.add_argument("--min_mem_tokens", type=int, default=512,
                    help="[long_context] min context tokens (curriculum floor).")
    ap.add_argument("--target_len", type=int, default=256,
                    help="[long_context] held-out continuation length to predict.")
    ap.add_argument("--enc_window", type=int, default=512,
                    help="[long_context] window size for the borrowed-encoder "
                         "pass so it never runs over an OOD context length.")
    ap.add_argument("--grad_checkpoint", action="store_true",
                    help="[long_context] enable decoder gradient checkpointing "
                         "(off by default; it produces NaN grads in bf16 with "
                         "inputs_embeds). Only needed to avoid OOM at very long "
                         "soft memories.")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fkl_weight", type=float, default=0.0,
                    help="weight on forward-KL to full-text next-token dist")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify")
    ap.add_argument("--save", default=os.path.join(
        rv.EXPERIMENT_DIR, "checkpoints", "softtoken.pt"))
    ap.add_argument("--init_ckpt", default=None,
                    help="warm-start the compressor from a previous checkpoint "
                         "before training (true two-stage training: e.g. stage 1 "
                         "reconstruction/long_context -> save, then stage 2 "
                         "--qa_train/--qa_recon --init_ckpt <stage1>). Loads the "
                         "trained head (adapter/gate/pool/encoder) via "
                         "comp.load_trained; the optimizer restarts fresh.")
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.n_chunks, args.max_len = 20, 8, 128

    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = rv.resolve_model(args.decoder)
    if os.path.isabs(path) and not os.path.exists(path):
        raise FileNotFoundError(
            f"Resolved decoder '{args.decoder}' to '{path}', but that path does "
            "not exist. Pass a valid local path or a Hugging Face model ID.")
    print(f"[st] loading decoder {path}")
    tok = AutoTokenizer.from_pretrained(path)
    decoder = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda")
    decoder.eval()

    comp = SoftTokenCompressor(
        decoder, factor=args.factor, enc_layers=args.enc_layers,
        train_encoder=args.train_encoder, mode=args.mode,
        role_factors={"user": args.user_factor,
                      "assistant": args.assistant_factor},
        pool_mode=args.pool).cuda()

    if args.init_ckpt:
        print(f"[st] warm-starting from {args.init_ckpt} (two-stage training)")
        comp.load_trained(args.init_ckpt)

    trainable = [p for p in comp.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[st] trainable params: {n_train/1e6:.2f}M "
          f"(factor={args.factor}, enc_layers={args.enc_layers})")
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    if args.long_context:
        _run_long_context_training(args, comp, decoder, tok, opt)
        _save_ckpt(args, comp)
        return

    if args.mix_manifest:
        _run_mix_manifest_training(args, comp, decoder, tok, opt)
        _save_ckpt(args, comp)
        return

    if args.qa_recon:
        _run_qa_recon_training(args, comp, decoder, tok, opt)
        _save_ckpt(args, comp)
        return

    if args.qa_train:
        _run_qa_training(args, comp, decoder, tok, opt)
        _save_ckpt(args, comp)
        return

    print(f"[st] building {args.n_chunks} chunks of {args.max_len} tokens")
    chunks, chunk_masks = make_chunks(args.dataset, args.data, tok, args.max_len, args.n_chunks)
    print(f"[st] got {len(chunks)} chunks")
    if not chunks:
        raise RuntimeError(
            f"No chunks were built from {args.data}. Check that the data exists "
            "and uses the expected schema."
        )

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

    _save_ckpt(args, comp)


if __name__ == "__main__":
    main()
