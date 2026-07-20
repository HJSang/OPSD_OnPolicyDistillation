#!/usr/bin/env python3
"""
Zero-training validation: does Visual-Text Compression (VTC) hurt conversational
memory more than text-based compression?

Core hypothesis (from our research discussion):
    Compression modality should match task structure. VTC works for dense long
    documents but FAILS on conversational memory, which needs precise retrieval
    of low-density user facts. We expect VTC to degrade most on precise-fact /
    multi-hop questions, while text summary / raw text hold up better.

This script runs NO training. It just compares, on LoCoMo memory QA:
    A. raw     -- full conversation as text            (upper bound)
    B. summary -- LLM-summarized conversation as text  (text compression)
    C. vtc     -- conversation rendered to images, fed to a VLM (visual compression)

and breaks accuracy down by question category, alongside the achieved
compression ratio for each condition.

Everything runs on a single GPU. Uses HuggingFace transformers (stable, well
documented API) rather than vLLM so the multimodal path is predictable.

First-run checklist:
    1. `pip install -r requirements.txt`
    2. Confirm the LoCoMo data URL below resolves, or pass --data_path to a local copy.
    3. Start with `--limit 5 --conditions raw` to smoke-test the text path,
       then add `summary` and `vtc`.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from io import BytesIO

# LoCoMo: 10 very-long multi-session dialogues with categorized memory QA.
# Category codes (from the LoCoMo paper):
#   1 = multi-hop, 2 = temporal reasoning, 3 = open-domain knowledge,
#   4 = single-hop, 5 = adversarial (answer not in conversation).
DEFAULT_LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

# --------------------------------------------------------------------------- #
# Portable model registry. Environment variables can point aliases at local
# mirrors; otherwise transformers downloads the public Hugging Face repository.
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {
    # ---- text readers ----
    "qwen2.5-7b": os.environ.get(
        "VTC_MODEL_QWEN2_5_7B", "Qwen/Qwen2.5-7B-Instruct"),
    "qwen3-4b": os.environ.get(
        "VTC_MODEL_QWEN3_4B", "Qwen/Qwen3-4B-Instruct-2507"),
    "qwen3.5-4b": os.environ.get(
        "VTC_MODEL_QWEN3_5_4B", "Qwen/Qwen3.5-4B"),
    "qwen3-8b": os.environ.get(
        "VTC_MODEL_QWEN3_8B", "Qwen/Qwen3-8B"),
    # ---- vision-language (VTC) ----
    "qwen2.5-vl-7b": os.environ.get(
        "VTC_MODEL_QWEN2_5_VL_7B", "Qwen/Qwen2.5-VL-7B-Instruct"),
    # ---- visual compressor (used by run_dsocr_reconstruct.py) ----
    "deepseek-ocr": os.environ.get(
        "VTC_MODEL_DEEPSEEK_OCR", "deepseek-ai/DeepSeek-OCR"),
}

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("VTC_DATA_DIR", os.path.join(EXPERIMENT_DIR, "data"))
RESULTS_DIR = os.environ.get(
    "VTC_RESULTS_DIR", os.path.join(EXPERIMENT_DIR, "results"))


def resolve_model(name_or_path):
    """Resolve a short name to a public model ID or environment override."""
    return os.path.expanduser(MODEL_REGISTRY.get(name_or_path, name_or_path))


def resolve_data_path(path):
    """Resolve relative data paths independently of the caller's cwd."""
    if re.match(r"https?://", path) or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(EXPERIMENT_DIR, path)


# --------------------------------------------------------------------------- #
# Data loading / parsing
# --------------------------------------------------------------------------- #
def load_json(data_path_or_url):
    data_path_or_url = resolve_data_path(data_path_or_url)
    if os.path.exists(data_path_or_url):
        with open(data_path_or_url) as f:
            return json.load(f)
    if not re.match(r"https?://", data_path_or_url):
        raise FileNotFoundError(
            f"Data file not found: {data_path_or_url}. Run "
            "`python prepare_data.py` in experiments/vtc_memory_validation, "
            "or pass a valid local path/URL."
        )
    print(f"[data] downloading from {data_path_or_url}")
    with urllib.request.urlopen(data_path_or_url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def conversation_to_text(sample):
    """Flatten a LoCoMo sample's multi-session conversation into a single string."""
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")
    lines = []
    # sessions are keyed session_1, session_2, ... with session_N_date_time siblings
    session_ids = sorted(
        [
            int(k.split("_")[1])
            for k in conv.keys()
            if re.fullmatch(r"session_\d+", k)
        ]
    )
    for sid in session_ids:
        key = f"session_{sid}"
        date = conv.get(f"{key}_date_time", "")
        lines.append(f"\n=== Session {sid} ({date}) ===")
        for turn in conv[key]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            # some turns carry image captions ("blip_caption"); include if present
            cap = turn.get("blip_caption")
            if cap:
                text = f"{text} [shared image: {cap}]"
            lines.append(f"{speaker}: {text}")
    header = f"Conversation between {speaker_a} and {speaker_b}.\n"
    return header + "\n".join(lines)


def iter_qa_locomo(dataset, limit_per_sample=None):
    """Yield (sample_index, conversation_text, question, gold_answer, category_name)."""
    for si, sample in enumerate(dataset):
        conv_text = conversation_to_text(sample)
        qa_list = sample.get("qa", [])
        n = 0
        for qa in qa_list:
            # category is stored as a string ("1".."5") in LoCoMo; coerce to int
            try:
                cat = int(qa.get("category"))
            except (TypeError, ValueError):
                cat = qa.get("category")
            cat_name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
            # adversarial (cat 5) items carry `adversarial_answer` instead of `answer`
            gold = qa.get("answer", qa.get("adversarial_answer", ""))
            question = qa.get("question", "")
            if not question:
                continue
            yield si, conv_text, question, str(gold), cat_name
            n += 1
            if limit_per_sample and n >= limit_per_sample:
                break


def longmemeval_to_text(instance):
    """Flatten a LongMemEval instance's haystack_sessions into a single string."""
    lines = []
    sessions = instance.get("haystack_sessions", [])
    dates = instance.get("haystack_dates", [""] * len(sessions))
    for si, session in enumerate(sessions):
        date = dates[si] if si < len(dates) else ""
        lines.append(f"\n=== Session {si + 1} ({date}) ===")
        for turn in session:
            role = turn.get("role", "")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def iter_qa_longmemeval(dataset, limit_per_sample=None):
    """Yield (index, conversation_text, question, gold_answer, category_name).

    Each LongMemEval instance is a single question with its own haystack, so
    index == instance index. Abstention questions (id endswith '_abs') get the
    'abstention' category.
    """
    for i, inst in enumerate(dataset):
        conv_text = longmemeval_to_text(inst)
        qid = inst.get("question_id", "")
        cat_name = "abstention" if qid.endswith("_abs") else inst.get(
            "question_type", "unknown")
        gold = str(inst.get("answer", ""))
        question = inst.get("question", "")
        if not question:
            continue
        yield i, conv_text, question, gold, cat_name


def iter_qa_ultrachat(dataset, limit_per_sample=None):
    """UltraChat is a TRAINING corpus (no QA). Yield one 'item' per conversation
    whose conv_text is the flattened dialogue; question/gold/category are empty.
    Used only by make_chunks / full-mode turn extraction during training."""
    for i, inst in enumerate(dataset):
        turns = inst.get("turns") or inst.get("messages") or []
        lines = [f"{t.get('role','user')}: {t.get('content','')}" for t in turns]
        conv_text = "\n".join(lines)
        yield i, conv_text, "", "", "train"


def iter_items(dataset_name, dataset, limit_per_sample=None):
    if dataset_name == "locomo":
        yield from iter_qa_locomo(dataset, limit_per_sample)
    elif dataset_name == "longmemeval":
        yield from iter_qa_longmemeval(dataset, limit_per_sample)
    elif dataset_name == "ultrachat":
        yield from iter_qa_ultrachat(dataset, limit_per_sample)
    else:
        raise ValueError(f"unknown dataset {dataset_name}")


# --------------------------------------------------------------------------- #
# Text rendering -> images (for the VTC condition)
# --------------------------------------------------------------------------- #
def render_text_to_images(text, font_path=None, font_size=16, page_w=1024,
                          page_h=1448, margin=24, line_spacing=4):
    """Render a long string into a list of PIL page images (monospace, wrapped)."""
    from PIL import Image, ImageDraw, ImageFont

    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        # DejaVuSansMono ships with matplotlib/PIL on most images
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # estimate chars per line from glyph width
    tmp = Image.new("RGB", (10, 10), "white")
    d = ImageDraw.Draw(tmp)
    char_w = max(1, d.textlength("M", font=font))
    max_chars = max(10, int((page_w - 2 * margin) / char_w))
    line_h = font_size + line_spacing
    lines_per_page = max(1, int((page_h - 2 * margin) / line_h))

    # wrap
    wrapped = []
    for raw_line in text.split("\n"):
        if raw_line == "":
            wrapped.append("")
            continue
        while len(raw_line) > max_chars:
            wrapped.append(raw_line[:max_chars])
            raw_line = raw_line[max_chars:]
        wrapped.append(raw_line)

    # paginate
    images = []
    for start in range(0, len(wrapped), lines_per_page):
        chunk = wrapped[start:start + lines_per_page]
        img = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(img)
        y = margin
        for ln in chunk:
            draw.text((margin, y), ln, fill="black", font=font)
            y += line_h
        images.append(img)
    return images


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class TextModel:
    def __init__(self, model_name, max_new_tokens=256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[text] loading {model_name}")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.max_new_tokens = max_new_tokens

    def count_tokens(self, text):
        return len(self.tok(text)["input_ids"])

    def chat(self, system, user, max_new_tokens=None):
        import torch

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


class VLModel:
    def __init__(self, model_name, max_new_tokens=256, max_pixels=None,
                 min_pixels=None):
        import torch
        from transformers import AutoProcessor

        print(f"[vl] loading {model_name} (max_pixels={max_pixels})")
        # Qwen2.5-VL class name; fall back to generic if unavailable
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLClass
        except Exception:
            from transformers import AutoModelForVision2Seq as VLClass
        # max_pixels/min_pixels control how much the processor downsamples each
        # page -> this is the real VTC compression knob (analogous to Glyph's DPI).
        proc_kwargs = {}
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels
        if min_pixels is not None:
            proc_kwargs["min_pixels"] = min_pixels
        self.processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)
        self.model = VLClass.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.max_new_tokens = max_new_tokens

    def count_image_tokens(self, images, text=""):
        """Number of tokens the processor assigns to the image(s)+text prompt."""
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=images, return_tensors="pt"
        )
        return int(inputs["input_ids"].shape[1])

    def chat(self, images, user_text, max_new_tokens=None):
        import torch

        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=images, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Prompts + judging
# --------------------------------------------------------------------------- #
QA_SYSTEM = (
    "You are answering a question about a long conversation. Answer concisely "
    "using ONLY information from the conversation. If the answer is not present, "
    "reply exactly: NOT MENTIONED."
)

SUMMARY_SYSTEM = (
    "You compress conversations into a dense factual summary that preserves every "
    "concrete fact either speaker stated about themselves (names, dates, "
    "preferences, events, relationships, numbers). Omit small talk. Be terse."
)

JUDGE_SYSTEM = (
    "You grade an answer against a gold answer for a memory question. "
    "Reply with exactly 'CORRECT' or 'WRONG'. Judge semantic equivalence, not "
    "wording. For questions with no answer in the source, 'NOT MENTIONED' style "
    "responses count as CORRECT only if the gold answer is also 'no information'."
)


def normalize(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def judge(text_model, question, gold, pred):
    # cheap exact-ish match first
    ng, np_ = normalize(gold), normalize(pred)
    if ng and (ng == np_ or ng in np_ or np_ in ng):
        return True
    verdict = text_model.chat(
        JUDGE_SYSTEM,
        f"Question: {question}\nGold answer: {gold}\nModel answer: {pred}\n"
        f"Grade (CORRECT/WRONG):",
        max_new_tokens=8,
    )
    return "CORRECT" in verdict.upper()


def make_summary(text_model, conv_text, target_ratio):
    """Compress the conversation to ~1/target_ratio of its token length."""
    n_tok = text_model.count_tokens(conv_text)
    budget = max(128, int(n_tok / target_ratio))
    summary = text_model.chat(
        SUMMARY_SYSTEM,
        f"Compress the following conversation into at most ~{budget} tokens, "
        f"preserving all concrete personal facts.\n\n{conv_text}",
        max_new_tokens=min(2048, budget + 128),
    )
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo",
                    choices=["locomo", "longmemeval"])
    ap.add_argument("--data_path", default=None,
                    help="Local path or URL to the dataset json. "
                         "Defaults: LoCoMo -> remote URL; "
                         "LongMemEval -> $VTC_DATA_DIR/longmemeval_oracle.json")
    ap.add_argument("--conditions", default="raw,summary,vtc",
                    help="Comma list subset of raw,summary,vtc,dsocr")
    ap.add_argument("--text_model", default="qwen2.5-7b",
                    help="Registry name (see MODEL_REGISTRY) or a "
                         "full path. e.g. qwen2.5-7b, qwen3-4b, qwen3.5-4b, qwen3-8b")
    ap.add_argument("--vl_model", default="qwen2.5-vl-7b",
                    help="Registry name or full path for the VTC model")
    ap.add_argument("--limit", type=int, default=30,
                    help="Max total QA items to evaluate")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip this many items after optional shuffling")
    ap.add_argument("--limit_per_sample", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle items before applying --limit (needed for "
                         "LongMemEval, which is grouped by question type)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary_ratio", type=float, default=4.0,
                    help="Target text-summary compression ratio")
    ap.add_argument("--font_size", type=int, default=16)
    ap.add_argument("--font_path", default=None)
    ap.add_argument("--vl_max_pixels", type=int, default=None,
                    help="Max pixels per rendered page for the VL processor "
                         "(the VTC compression knob; lower -> more compression, "
                         "fewer visual tokens). e.g. 200704 = 256*28*28.")
    ap.add_argument("--dsocr_cache", default="dsocr_cache.json",
                    help="Path to the DeepSeek-OCR reconstruction cache produced "
                         "by run_dsocr_reconstruct.py (used by the 'dsocr' condition)")
    ap.add_argument("--skip_judge", action="store_true",
                    help="only generate predictions; score them later with "
                         "official_longmemeval_judge.py")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "results.json"))
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    # resolve default data path per dataset
    data_path = args.data_path
    if data_path is None:
        data_path = (DEFAULT_LOCOMO_URL if args.dataset == "locomo"
                     else os.path.join(DATA_DIR, "longmemeval_oracle.json"))
    dataset = load_json(data_path)

    items = list(iter_items(args.dataset, dataset, args.limit_per_sample))
    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(items)
    if args.offset:
        items = items[args.offset:]
    if args.limit:
        items = items[: args.limit]
    print(f"[data] {args.dataset}: {len(items)} QA items "
          f"(from {len(dataset)} instances)")

    text_model = TextModel(resolve_model(args.text_model))
    vl_model = (VLModel(resolve_model(args.vl_model),
                        max_pixels=args.vl_max_pixels)
                if "vtc" in conditions else None)

    dsocr_cache = {}
    if "dsocr" in conditions:
        with open(args.dsocr_cache) as f:
            dsocr_cache = json.load(f)["cache"]
        print(f"[dsocr] loaded {len(dsocr_cache)} cached reconstructions "
              f"from {args.dsocr_cache}")

    # cache per-conversation artifacts so we don't recompute summary/images per QA
    summary_cache = {}
    image_cache = {}
    ratio_stats = defaultdict(list)  # condition -> list of compression ratios

    # results[condition][category] = [correct_bools]
    results = {c: defaultdict(list) for c in conditions}
    records = []

    for idx, (si, conv_text, question, gold, cat_name) in enumerate(items):
        full_tok = text_model.count_tokens(conv_text)
        qid = (
            dataset[si].get("question_id")
            if args.dataset == "longmemeval" else None
        )
        rec = {"i": idx, "sample": si, "category": cat_name,
               "question": question, "gold": gold, "full_tokens": full_tok}
        if qid:
            rec["question_id"] = qid

        if "raw" in conditions:
            pred = text_model.chat(QA_SYSTEM, f"{conv_text}\n\nQuestion: {question}")
            ok = None if args.skip_judge else judge(
                text_model, question, gold, pred)
            if ok is not None:
                results["raw"][cat_name].append(ok)
            ratio_stats["raw"].append(1.0)
            rec["raw_pred"], rec["raw_ok"] = pred, ok

        if "summary" in conditions:
            if si not in summary_cache:
                summary_cache[si] = make_summary(text_model, conv_text,
                                                 args.summary_ratio)
            summ = summary_cache[si]
            summ_tok = text_model.count_tokens(summ)
            ratio = full_tok / max(1, summ_tok)
            pred = text_model.chat(QA_SYSTEM, f"{summ}\n\nQuestion: {question}")
            ok = None if args.skip_judge else judge(
                text_model, question, gold, pred)
            if ok is not None:
                results["summary"][cat_name].append(ok)
            ratio_stats["summary"].append(ratio)
            rec["summary_pred"], rec["summary_ok"] = pred, ok
            rec["summary_tokens"] = summ_tok

        if "vtc" in conditions:
            if si not in image_cache:
                image_cache[si] = render_text_to_images(
                    conv_text, font_path=args.font_path, font_size=args.font_size
                )
            images = image_cache[si]
            img_tok = vl_model.count_image_tokens(images, question)
            ratio = full_tok / max(1, img_tok)
            pred = vl_model.chat(images, f"Question: {question}")
            ok = None if args.skip_judge else judge(
                text_model, question, gold, pred)
            if ok is not None:
                results["vtc"][cat_name].append(ok)
            ratio_stats["vtc"].append(ratio)
            rec["vtc_pred"], rec["vtc_ok"] = pred, ok
            rec["vtc_tokens"], rec["vtc_pages"] = img_tok, len(images)

        if "dsocr" in conditions:
            key = f"{args.dataset}:{si}"
            entry = dsocr_cache.get(key)
            if entry is None:
                raise KeyError(
                    f"no DeepSeek-OCR cache entry for {key}; run "
                    f"run_dsocr_reconstruct.py with matching "
                    f"--dataset/--limit/--shuffle/--seed first")
            recon = entry["reconstructed"]
            # compressed size = vision tokens (assistant) + verbatim user tokens
            vtok = entry["vision_tokens"] + entry.get("text_tokens", 0)
            ratio = full_tok / max(1, vtok)
            pred = text_model.chat(QA_SYSTEM, f"{recon}\n\nQuestion: {question}")
            ok = None if args.skip_judge else judge(
                text_model, question, gold, pred)
            if ok is not None:
                results["dsocr"][cat_name].append(ok)
            ratio_stats["dsocr"].append(ratio)
            rec["dsocr_pred"], rec["dsocr_ok"] = pred, ok
            rec["dsocr_tokens"], rec["dsocr_pages"] = vtok, entry["pages"]

        records.append(rec)
        if (idx + 1) % 5 == 0:
            print(f"  ... {idx + 1}/{len(items)} done")

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    def acc(bools):
        return sum(bools) / len(bools) if bools else float("nan")

    print("\n================ RESULTS ================")
    if args.skip_judge:
        print("Judging skipped; run official_longmemeval_judge.py on this output.")
    all_cats = sorted({c for cond in results for c in results[cond]})
    header = f"{'category':<14}" + "".join(f"{c:>12}" for c in conditions)
    print(header)
    for cat_name in all_cats:
        row = f"{cat_name:<14}"
        for c in conditions:
            row += f"{acc(results[c][cat_name]):>12.3f}"
        print(row)
    # overall
    row = f"{'OVERALL':<14}"
    for c in conditions:
        allb = [b for cat in results[c].values() for b in cat]
        row += f"{acc(allb):>12.3f}"
    print(row)

    print("\n---- mean compression ratio ----")
    for c in conditions:
        r = ratio_stats[c]
        print(f"  {c:<10} {sum(r)/len(r):.2f}x" if r else f"  {c}: n/a")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "records": records}, f, indent=2)
    print(f"\n[out] wrote per-item records to {args.out}")


if __name__ == "__main__":
    main()
