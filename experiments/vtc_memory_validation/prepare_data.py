#!/usr/bin/env python3
"""
Download the benchmarks used by the VTC-vs-text conversational-memory validation.

    LoCoMo       -> data/locomo10.json                  (2.8 MB, 10 dialogues, 1986 QA)
    LongMemEval  -> data/longmemeval_oracle.json         (15 MB, 500 QA, oracle sessions)
                    data/longmemeval_s_cleaned.json      (277 MB, optional, full haystack)
    UltraChat    -> data/ultrachat_train.json            (2,000-conversation train_sft subset by default)

Run in any environment with network access:
    python prepare_data.py                 # LoCoMo + LongMemEval oracle + UltraChat subset
    python prepare_data.py --with_s        # also the 277 MB _s file
"""
import argparse
import json
import os
import shutil
import urllib.request

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LME_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
LME_FILES = {
    "longmemeval_oracle.json": f"{LME_BASE}/longmemeval_oracle.json",
    "longmemeval_s_cleaned.json": f"{LME_BASE}/longmemeval_s_cleaned.json",
}
ULTRACHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
BUNDLED_ULTRACHAT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "longmemeval_evaluation_training_data",
    "ultrachat_train.json",
)
DEFAULT_DATA_DIR = os.environ.get(
    "VTC_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"[skip] {dest} already exists ({os.path.getsize(dest)} bytes)")
        return
    print(f"[get ] {url}\n    -> {dest}")
    tmp = dest + ".tmp"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    print(f"[done] {dest} ({os.path.getsize(dest)} bytes)")


def normalize_ultrachat_row(row, fallback_id):
    turns = []
    for msg in row.get("messages") or []:
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        role = msg.get("role", "user")
        role = "user" if role in ("user", "human") else "assistant"
        turns.append({"role": role, "content": content})
    if len(turns) < 2:
        return None
    return {"id": row.get("prompt_id") or fallback_id, "turns": turns}


def download_ultrachat(dest, split, limit, seed, shuffle_buffer):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"[skip] {dest} already exists ({os.path.getsize(dest)} bytes)")
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "UltraChat preparation requires the `datasets` package. "
            "Run `pip install -r requirements.txt`, or pass --skip_ultrachat."
        ) from exc

    print(f"[get ] {ULTRACHAT_DATASET} split={split} subset_size={limit}\n    -> {dest}")
    ds = load_dataset(ULTRACHAT_DATASET, split=split, streaming=True)
    if shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    rows = []
    for row in ds:
        item = normalize_ultrachat_row(row, f"{split}-{len(rows)}")
        if item:
            rows.append(item)
        if len(rows) >= limit:
            break
    if len(rows) < limit:
        raise RuntimeError(
            f"Only collected {len(rows)} UltraChat conversations; expected {limit}."
        )

    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    os.replace(tmp, dest)
    print(f"[done] {dest} ({len(rows)} sampled conversations, {os.path.getsize(dest)} bytes)")


def prepare_ultrachat(dest, split, limit, seed, shuffle_buffer, refresh=False):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"[skip] {dest} already exists ({os.path.getsize(dest)} bytes)")
        return
    if not refresh and os.path.exists(BUNDLED_ULTRACHAT):
        shutil.copy2(BUNDLED_ULTRACHAT, dest)
        print(f"[copy] bundled UltraChat subset {BUNDLED_ULTRACHAT}\n    -> {dest}")
        return
    download_ultrachat(dest, split, limit, seed, shuffle_buffer)


def stage_files(data_dir, stage_dir):
    os.makedirs(stage_dir, exist_ok=True)
    required = [
        "ultrachat_train.json",
        "longmemeval_oracle.json",
    ]
    optional = [
        "locomo10.json",
        "longmemeval_s_cleaned.json",
    ]
    missing = [name for name in required
               if not os.path.exists(os.path.join(data_dir, name))]
    if missing:
        raise FileNotFoundError(
            "Cannot stage data; missing required files in "
            f"{os.path.abspath(data_dir)}: {', '.join(missing)}"
        )
    for name in required + optional:
        src = os.path.join(data_dir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(stage_dir, name)
        shutil.copy2(src, dst)
        print(f"[stage] {src} -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--with_s", action="store_true",
                    help="also download the 277 MB longmemeval_s_cleaned.json")
    ap.add_argument("--skip_ultrachat", action="store_true",
                    help="skip data/ultrachat_train.json generation")
    ap.add_argument("--refresh_ultrachat", action="store_true",
                    help="download a fresh UltraChat subset instead of using the "
                         "bundled tracked subset when available")
    ap.add_argument("--ultrachat_split", default="train_sft")
    ap.add_argument("--ultrachat_limit", type=int, default=2000)
    ap.add_argument("--ultrachat_seed", type=int, default=0)
    ap.add_argument("--ultrachat_shuffle_buffer", type=int, default=10000)
    ap.add_argument("--stage_dir", default=None,
                    help="optional second directory to copy prepared data into")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    download(LOCOMO_URL, os.path.join(args.data_dir, "locomo10.json"))
    download(LME_FILES["longmemeval_oracle.json"],
             os.path.join(args.data_dir, "longmemeval_oracle.json"))
    if args.with_s:
        download(LME_FILES["longmemeval_s_cleaned.json"],
                 os.path.join(args.data_dir, "longmemeval_s_cleaned.json"))
    if not args.skip_ultrachat:
        prepare_ultrachat(
            os.path.join(args.data_dir, "ultrachat_train.json"),
            args.ultrachat_split,
            args.ultrachat_limit,
            args.ultrachat_seed,
            args.ultrachat_shuffle_buffer,
            args.refresh_ultrachat,
        )
    if args.stage_dir:
        stage_files(args.data_dir, args.stage_dir)
    print("\nAll set. Data in:", os.path.abspath(args.data_dir))


if __name__ == "__main__":
    main()
