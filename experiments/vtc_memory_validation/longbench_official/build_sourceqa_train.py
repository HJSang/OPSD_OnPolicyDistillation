#!/usr/bin/env python3
"""Build source-task QA training data for LongBench task-adapted compression.

Output format matches ``softtoken/train.py --qa_train``:
``[{ "source": ..., "passage": ..., "qas": [{"q": ..., "a": ...}] }]``.

This uses source-task train splits where available. MultiFieldQA is intentionally
excluded because LongBench's MultiFieldQA is test-only / author-annotated.
"""
import argparse
import json
import os
import random
import tarfile
import tempfile
import urllib.request
from collections import defaultdict


QASPER_TRAIN_DEV_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
    "qasper-train-dev-v0.3.tgz"
)


def clip(text, max_chars):
    text = " ".join(str(text).split())
    return text[:max_chars] if max_chars and len(text) > max_chars else text


def add_item(items, source, passage, qas, max_chars):
    passage = clip(passage, max_chars)
    clean_qas = []
    for qa in qas:
        q = " ".join(str(qa.get("q", "")).split())
        a = " ".join(str(qa.get("a", "")).split())
        if q and a:
            clean_qas.append({"q": q, "a": a})
    if passage and clean_qas:
        items.append({"source": source, "passage": passage, "qas": clean_qas})


def cap_by_qas(items, max_qas, seed):
    if not max_qas:
        return items
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    capped = []
    n_qas = 0
    for item in items:
        qas = list(item["qas"])
        rng.shuffle(qas)
        remaining = max_qas - n_qas
        if remaining <= 0:
            break
        qas = qas[:remaining]
        if qas:
            capped.append({**item, "qas": qas})
            n_qas += len(qas)
    return capped


def build_narrativeqa(max_docs, max_qas, max_chars, seed):
    from datasets import load_dataset

    ds = load_dataset("deepmind/narrativeqa", split="train")
    by_doc = defaultdict(lambda: {"passage": "", "qas": []})
    for row in ds:
        doc = row["document"]
        did = doc["id"]
        # The public train split exposes summaries, not full book/script text.
        by_doc[did]["passage"] = doc.get("summary", {}).get("text", "")
        answers = row.get("answers") or []
        answer = answers[0].get("text", "") if answers else ""
        by_doc[did]["qas"].append({"q": row["question"]["text"], "a": answer})
    items = []
    for v in by_doc.values():
        add_item(items, "narrativeqa_train", v["passage"], v["qas"], max_chars)
    items = cap_by_qas(items, max_qas, seed)
    return items[:max_docs] if max_docs else items


def qasper_answer(answer_obj):
    if answer_obj.get("unanswerable"):
        return ""
    if answer_obj.get("free_form_answer"):
        return answer_obj["free_form_answer"]
    spans = answer_obj.get("extractive_spans") or []
    if spans:
        return "; ".join(spans)
    yes_no = answer_obj.get("yes_no")
    if yes_no is True:
        return "Yes"
    if yes_no is False:
        return "No"
    return ""


def build_qasper(max_docs, max_qas, max_chars, seed):
    with tempfile.TemporaryDirectory() as td:
        archive = os.path.join(td, "qasper-train-dev-v0.3.tgz")
        urllib.request.urlretrieve(QASPER_TRAIN_DEV_URL, archive)
        with tarfile.open(archive) as tar:
            member = next(
                m for m in tar.getmembers()
                if m.name.endswith("qasper-train-v0.3.json")
            )
            raw = json.load(tar.extractfile(member))

    items = []
    for paper in raw.values():
        parts = [paper.get("title", ""), paper.get("abstract", "")]
        for section in paper.get("full_text", []):
            name = section.get("section_name") or ""
            if name:
                parts.append(name)
            parts.extend(section.get("paragraphs") or [])
        qas = []
        for qa in paper.get("qas", []):
            answer = ""
            for ann in qa.get("answers", []):
                answer = qasper_answer(ann.get("answer", {}))
                if answer:
                    break
            if answer:
                qas.append({"q": qa.get("question", ""), "a": answer})
        add_item(items, "qasper_train", "\n\n".join(parts), qas, max_chars)
    items = cap_by_qas(items, max_qas, seed)
    return items[:max_docs] if max_docs else items


def join_hotpot_context(context):
    titles = context["title"]
    sentences = context["sentences"]
    return "\n\n".join(
        f"{title}\n" + " ".join(sents)
        for title, sents in zip(titles, sentences)
    )


def build_hotpotqa(max_docs, max_qas, max_chars, seed):
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    items = []
    for row in rows:
        add_item(
            items,
            "hotpotqa_train",
            join_hotpot_context(row["context"]),
            [{"q": row["question"], "a": row["answer"]}],
            max_chars,
        )
        limit = max_qas or max_docs
        if limit and sum(len(x["qas"]) for x in items) >= limit:
            break
    return items[:max_docs] if max_docs else items


def build_2wikimqa(max_docs, max_qas, max_chars, seed):
    from datasets import load_dataset

    ds = load_dataset("voidful/2WikiMultihopQA", split="train")
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    items = []
    for row in rows:
        passage = "\n\n".join(
            f"{title}\n" + " ".join(paragraphs)
            for title, paragraphs in row["context"]
        )
        add_item(
            items,
            "2wikimqa_train",
            passage,
            [{"q": row["question"], "a": row["answer"]}],
            max_chars,
        )
        limit = max_qas or max_docs
        if limit and sum(len(x["qas"]) for x in items) >= limit:
            break
    return items[:max_docs] if max_docs else items


def build_musique(max_docs, max_qas, max_chars, seed):
    from datasets import load_dataset

    ds = load_dataset("dgslibisey/MuSiQue", split="train")
    rows = [r for r in ds if r.get("answerable", True)]
    random.Random(seed).shuffle(rows)
    items = []
    for row in rows:
        passage = "\n\n".join(
            f"{p['title']}\n{p['paragraph_text']}"
            for p in row["paragraphs"]
        )
        add_item(
            items,
            "musique_train",
            passage,
            [{"q": row["question"], "a": row["answer"]}],
            max_chars,
        )
        limit = max_qas or max_docs
        if limit and sum(len(x["qas"]) for x in items) >= limit:
            break
    return items[:max_docs] if max_docs else items


BUILDERS = [
    ("narrativeqa", build_narrativeqa),
    ("qasper", build_qasper),
    ("hotpotqa", build_hotpotqa),
    ("2wikimqa", build_2wikimqa),
    ("musique", build_musique),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_docs_per_source", type=int, default=None)
    ap.add_argument("--max_qas_per_source", type=int, default=2000)
    ap.add_argument("--max_chars", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_items = []
    summary = {}
    for name, builder in BUILDERS:
        items = builder(
            args.max_docs_per_source,
            args.max_qas_per_source,
            args.max_chars,
            args.seed,
        )
        all_items.extend(items)
        summary[name] = {
            "passages": len(items),
            "qas": sum(len(x["qas"]) for x in items),
        }
        print(f"[sourceqa] {name}: {summary[name]}")

    random.Random(args.seed).shuffle(all_items)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False)
    print(f"[sourceqa] wrote {len(all_items)} passages to {args.out}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
