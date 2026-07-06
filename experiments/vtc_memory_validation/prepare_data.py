#!/usr/bin/env python3
"""
Download the benchmarks used by the VTC-vs-text conversational-memory validation.

    LoCoMo       -> data/locomo10.json                  (2.8 MB, 10 dialogues, 1986 QA)
    LongMemEval  -> data/longmemeval_oracle.json         (15 MB, 500 QA, oracle sessions)
                    data/longmemeval_s_cleaned.json      (277 MB, optional, full haystack)

Run on the GPU pod (net access required):
    python prepare_data.py                 # LoCoMo + LongMemEval oracle
    python prepare_data.py --with_s        # also the 277 MB _s file
"""
import argparse
import os
import urllib.request

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LME_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
LME_FILES = {
    "longmemeval_oracle.json": f"{LME_BASE}/longmemeval_oracle.json",
    "longmemeval_s_cleaned.json": f"{LME_BASE}/longmemeval_s_cleaned.json",
}


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


def build_ultrachat(data_dir, n_convs=2000):
    """Download an UltraChat shard and write n_convs conversations as JSON
    (`turns` = list of {role, content}) for encoder training."""
    dest = os.path.join(data_dir, "ultrachat_train.json")
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"[skip] {dest} already exists")
        return
    import pandas as pd
    from huggingface_hub import hf_hub_download
    print("[get ] UltraChat 200k (sft shard)")
    p = hf_hub_download(
        repo_id="HuggingFaceH4/ultrachat_200k", repo_type="dataset",
        filename="data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
        local_dir=os.path.join(data_dir, "_ultrachat_raw"))
    df = pd.read_parquet(p)
    out = []
    for i in range(min(n_convs, len(df))):
        msgs = df.iloc[i]["messages"]
        turns = [{"role": m["role"], "content": m["content"]}
                 for m in msgs if m["role"] in ("user", "assistant")]
        if len(turns) >= 2:
            out.append({"conversation_id": f"uc_{i}", "turns": turns})
    with open(dest, "w") as f:
        json.dump(out, f)
    print(f"[done] {dest} ({len(out)} conversations)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--with_s", action="store_true",
                    help="also download the 277 MB longmemeval_s_cleaned.json")
    ap.add_argument("--ultrachat", action="store_true",
                    help="build the UltraChat training corpus (encoder training)")
    ap.add_argument("--n_ultrachat", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    if args.ultrachat:
        build_ultrachat(args.data_dir, args.n_ultrachat)
        return
    download(LOCOMO_URL, os.path.join(args.data_dir, "locomo10.json"))
    download(LME_FILES["longmemeval_oracle.json"],
             os.path.join(args.data_dir, "longmemeval_oracle.json"))
    if args.with_s:
        download(LME_FILES["longmemeval_s_cleaned.json"],
                 os.path.join(args.data_dir, "longmemeval_s_cleaned.json"))
    print("\nAll set. Data in:", os.path.abspath(args.data_dir))


if __name__ == "__main__":
    main()
