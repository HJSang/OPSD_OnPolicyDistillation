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

Example:
  python softtoken/eval_softtoken.py \
      --ckpt checkpoints/softtoken_simple_f8.pt --limit 20

The compressor architecture is restored from the training arguments embedded
in the checkpoint. CLI options can override those values for ablations.
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


QA_INSTR_ABSTAIN = ("\n\nAnswer the question using only the conversation above. "
                    "If not present, reply: NOT MENTIONED.\nQuestion: ")
QA_INSTR_NO_ABSTAIN = ("\n\nAnswer the question using only the conversation above. "
                       "Give a short answer.\nQuestion: ")

# When answering from soft tokens the reader is fed a bare text continuation
# ([soft ; question] with no chat template), so it runs in transcript-completion
# mode and frequently keeps decoding past its answer, hallucinating fake
# follow-up turns ("\nuser: ...\nAnswer: ..."). Those leaked turns are not part
# of the answer but drag an LLM judge toward "no". We cut the response at the
# first leaked-turn marker so only the actual answer is scored. This recovers
# ~0.10-0.12 accuracy on LongMemEval and is the honest decoding boundary.
LEAK_MARKERS = ("\nuser:", "\nUser:", "\nUSER:", "\nassistant:", "\nAssistant:",
                "\nAnswer:", "\nQuestion:", "\nQ:", "\nA:", "\nB:")


def strip_leaked_turns(text):
    """Truncate a soft-token answer at the first hallucinated follow-up turn."""
    for sep in LEAK_MARKERS:
        text = text.split(sep)[0]
    return text.strip()


def generate_from_soft(comp, tok, soft, question, qa_instr, max_new_tokens=64):
    """soft: (1, M, d). Feed [soft ; question_emb] and greedily decode."""
    q_ids = tok(qa_instr + question + "\nAnswer:", return_tensors="pt")[
        "input_ids"].to(soft.device)
    q_emb = comp.embed_tokens(q_ids)
    inp = torch.cat([soft, q_emb], dim=1)
    attn = torch.ones(inp.shape[:2], dtype=torch.long, device=inp.device)
    eos_ids = [i for i in (tok.eos_token_id,
                           tok.convert_tokens_to_ids("<|im_end|>"))
               if isinstance(i, int) and i >= 0]
    with torch.no_grad():
        out = comp.decoder.generate(
            inputs_embeds=inp, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=eos_ids or None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True).strip()
    return strip_leaked_turns(text)


def generate_from_full_text(decoder, tok, conversation, question, qa_instr,
                            max_new_tokens=64):
    """Uncompressed readability control: feed full text tokens directly."""
    prompt = conversation + qa_instr + question + "\nAnswer:"
    ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
    eos_ids = [i for i in (tok.eos_token_id,
                           tok.convert_tokens_to_ids("<|im_end|>"))
               if isinstance(i, int) and i >= 0]
    with torch.no_grad():
        out = decoder.generate(
            input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=eos_ids or None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return strip_leaked_turns(text)


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


def _locomo_conversation_to_ab_text(sample):
    """Flatten LoCoMo with eval-time speaker names normalized to A/B.

    SynthLoCoMo trains on `A:`/`B:` passage turns and questions phrased as
    `Speaker A/B`. This transform tests whether real-name surface forms are a
    refusal trigger without changing the checkpoint or LoCoMo content.
    """
    import re
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")
    lines = ["Conversation between Speaker A and Speaker B."]
    session_ids = sorted(
        int(k.split("_")[1]) for k in conv
        if re.fullmatch(r"session_\d+", k))
    for sid in session_ids:
        key = f"session_{sid}"
        date = conv.get(f"{key}_date_time", "")
        lines.append(f"\n=== Session {sid} ({date}) ===")
        for turn in conv[key]:
            speaker = turn.get("speaker", "")
            if speaker == speaker_a:
                label = "A"
            elif speaker == speaker_b:
                label = "B"
            else:
                label = speaker
            text = turn.get("text", "")
            cap = turn.get("blip_caption")
            if cap:
                text = f"{text} [shared image: {cap}]"
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _normalize_locomo_question(question, sample):
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "")
    speaker_b = conv.get("speaker_b", "")
    if speaker_a:
        question = question.replace(speaker_a, "Speaker A")
    if speaker_b:
        question = question.replace(speaker_b, "Speaker B")
    return question


def _iter_locomo_ab_items(raw):
    for si, sample in enumerate(raw):
        conv_text = _locomo_conversation_to_ab_text(sample)
        qa_list = sample.get("qa", [])
        for qa in qa_list:
            try:
                cat = int(qa.get("category"))
            except (TypeError, ValueError):
                cat = qa.get("category")
            cat_name = rv.CATEGORY_NAMES.get(cat, f"cat_{cat}")
            gold = qa.get("answer", qa.get("adversarial_answer", ""))
            question = _normalize_locomo_question(qa.get("question", ""), sample)
            if question:
                yield si, conv_text, question, str(gold), cat_name


def _iter_eval_items(dataset, raw, normalize_locomo_speakers=False):
    if dataset == "synthlocomo":
        for si, item in enumerate(raw):
            passage = item.get("passage", "")
            for qa in item.get("qas", []):
                q, a = qa.get("q", ""), qa.get("a", "")
                if passage and q and a:
                    yield si, passage, q, str(a), qa.get("category", "unknown")
        return
    if dataset == "locomo" and normalize_locomo_speakers:
        yield from _iter_locomo_ab_items(raw)
        return
    yield from rv.iter_items(dataset, raw, limit_per_sample=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--factor", type=int, default=None,
                    help="override the checkpoint's compression factor")
    ap.add_argument("--decoder", default=None,
                    help="override the checkpoint's decoder/reader")
    ap.add_argument("--mode", choices=["simple", "full"], default=None)
    ap.add_argument("--user_factor", type=int, default=None)
    ap.add_argument("--assistant_factor", type=int, default=None)
    ap.add_argument("--enc_layers", type=int, default=None)
    ap.add_argument("--pool", choices=["mean", "attn", "attn4"], default=None)
    ap.add_argument("--dataset", default="longmemeval")
    ap.add_argument("--data", default=os.path.join(
        rv.DATA_DIR, "longmemeval_oracle.json"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle before --limit (match run_validation for a "
                         "fair same-sample comparison on LongMemEval)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enc_window", type=int, default=0,
                    help="if >0, encode the conversation in windows of this many "
                         "tokens (simple mode). Match the value used at training "
                         "with --long_context so the borrowed encoder stays "
                         "in-distribution on long conversations.")
    ap.add_argument("--normalize_locomo_speakers", action="store_true",
                    help="eval-only ablation: map LoCoMo speaker names to A/B in "
                         "the passage and Speaker A/B in the question.")
    ap.add_argument("--no_abstain", action="store_true",
                    help="eval-only ablation: remove the NOT MENTIONED "
                         "instruction from the QA prompt.")
    ap.add_argument("--full_text", action="store_true",
                    help="eval-only readability control: answer from the full "
                         "uncompressed text instead of soft tokens.")
    ap.add_argument("--skip_judge", action="store_true",
                    help="only generate predictions; score them later with "
                         "official_longmemeval_judge.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    ck_args = ck.get("args", {}) if isinstance(ck, dict) else {}
    mode = args.mode or ck_args.get("mode", "simple")
    factor = args.factor or ck_args.get("factor", 8)
    enc_layers = args.enc_layers or ck_args.get("enc_layers", 2)
    user_factor = args.user_factor or ck_args.get("user_factor", 4)
    assistant_factor = args.assistant_factor or ck_args.get("assistant_factor", 16)
    dataset = args.dataset
    data = args.data

    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = rv.resolve_model(args.decoder or ck_args.get("decoder", "qwen2.5-7b"))
    print(f"[eval] loading decoder {path}")
    tok = AutoTokenizer.from_pretrained(path)
    decoder = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    # Match the encoder config to how the checkpoint was trained: if the ckpt
    # has no saved encoder layers it was trained with train_encoder=False (the
    # long-context recipe), so the borrowed layers must stay in bf16 to match.
    train_encoder = "enc_layers" in ck if isinstance(ck, dict) else True
    if isinstance(ck_args, dict) and "train_encoder" in ck_args:
        train_encoder = bool(ck_args["train_encoder"])
    # Reconstruct the token pooling used at training. If the ckpt was trained
    # with attention pooling it carries pool_key/pool_query; the compressor must
    # be built with pool_mode="attn" or load_trained silently drops them and
    # eval falls back to mean pooling (wrong, degraded results).
    pool_mode = args.pool or (ck_args.get("pool") if isinstance(ck_args, dict)
                              else None)
    if not pool_mode:
        pool_mode = "attn4" if (isinstance(ck, dict) and "pool_keys" in ck) \
            else "attn" if (isinstance(ck, dict) and "pool_key" in ck) \
            else "mean"
    del ck

    comp = SoftTokenCompressor(
        decoder, factor=factor,
        enc_layers=enc_layers, train_encoder=train_encoder,
        mode=mode,
        role_factors={"user": user_factor, "assistant": assistant_factor},
        pool_mode=pool_mode).cuda()
    comp.load_trained(args.ckpt)
    comp.eval()
    print(f"[eval] loaded ckpt {args.ckpt} (mode={mode}, "
          f"pool={pool_mode}, train_encoder={train_encoder})")

    raw = rv.load_json(data)
    items = list(_iter_eval_items(
        dataset, raw, normalize_locomo_speakers=args.normalize_locomo_speakers))
    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(items)
    if args.limit:
        items = items[: args.limit]
    print(f"[eval] {len(items)} QA items")

    results = defaultdict(list)
    ratios = []
    records = []
    qa_instr = QA_INSTR_NO_ABSTAIN if args.no_abstain else QA_INSTR_ABSTAIN

    for idx, (si, conv_text, question, gold, cat) in enumerate(items):
        full_ids = tok(conv_text, return_tensors="pt")["input_ids"].cuda()
        n_tok = full_ids.shape[1]

        if args.full_text:
            n_soft = n_tok
            pred = generate_from_full_text(
                decoder, tok, conv_text, question, qa_instr)
        else:
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
                    if args.enc_window and args.enc_window > 0:
                        soft = comp.encode_long(full_ids, mask,
                                                window=args.enc_window)  # (1, M, d)
                    else:
                        soft = comp.encode(full_ids, mask)  # (1, M, d)
                    n_soft = soft.shape[1]

            pred = generate_from_soft(
                comp, tok, soft.to(decoder.dtype), question, qa_instr)
        ok = None if args.skip_judge else judge_simple(
            tok, comp, question, gold, pred)
        if ok is not None:
            results[cat].append(ok)
        ratios.append(n_tok / max(1, n_soft))
        record = {"i": idx, "sample": si, "category": cat, "question": question,
                  "gold": gold, "pred": pred, "ok": ok,
                  "tokens": n_tok, "soft_tokens": n_soft}
        if dataset == "longmemeval":
            record["question_id"] = raw[si].get("question_id")
        records.append(record)
        if (idx + 1) % 5 == 0:
            print(f"  ... {idx + 1}/{len(items)}")

    print("\n============ SOFTTOKEN RESULTS ============")
    if args.skip_judge:
        print("Judging skipped; run official_longmemeval_judge.py on this output.")
    for cat in sorted(results):
        b = results[cat]
        print(f"{cat:<16} {sum(b)/len(b):.3f}  (n={len(b)})")
    allb = [x for v in results.values() for x in v]
    if allb:
        print(f"{'OVERALL':<16} {sum(allb)/len(allb):.3f}")
    print(f"mean compression: {sum(ratios)/len(ratios):.2f}x")

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    out = args.out or os.path.join(
        os.environ.get("VTC_RESULTS_DIR",
                       os.path.join(rv.EXPERIMENT_DIR, "results")),
        f"results_{ckpt_name}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "checkpoint_args": ck_args,
            "eval_args": vars(args),
            "ckpt": args.ckpt,
            "records": records,
        }, f, indent=2)
    print(f"[eval] wrote {out}")


def _turns_for_sample(dataset_name, raw, si):
    if dataset_name == "synthlocomo":
        turns = []
        for session in raw[si].get("sessions", []):
            for t in session.get("turns", []):
                turns.append({"role": t.get("speaker", "A"),
                              "content": t.get("text", "")})
        return turns
    if dataset_name == "longmemeval":
        inst = raw[si]
        return [{"role": t.get("role", "user"), "content": t.get("content", "")}
                for s in inst.get("haystack_sessions", []) for t in s]
    if dataset_name == "msc_lme":
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
