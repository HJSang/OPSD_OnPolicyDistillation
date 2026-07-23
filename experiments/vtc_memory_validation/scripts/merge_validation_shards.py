#!/usr/bin/env python3
"""Merge ordered run_validation.py shards into one result payload."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()

    records = []
    first_args = None
    comparable_args = None
    expected_offset = 0
    for shard in args.shards:
        with shard.open(encoding="utf-8") as f:
            payload = json.load(f)
        shard_args = payload["args"]
        shard_records = payload["records"]
        if shard_args.get("offset", 0) != expected_offset:
            raise ValueError(
                f"{shard}: expected offset {expected_offset}, "
                f"got {shard_args.get('offset', 0)}"
            )
        current_comparable = {
            key: value for key, value in shard_args.items()
            if key not in {"limit", "offset", "out"}
        }
        if first_args is None:
            first_args = dict(shard_args)
            comparable_args = current_comparable
        elif current_comparable != comparable_args:
            raise ValueError(f"{shard}: arguments differ from the first shard")
        records.extend(shard_records)
        expected_offset += len(shard_records)

    question_ids = [
        record.get("question_id") for record in records
        if record.get("question_id")
    ]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("merged records contain duplicate question IDs")

    for index, record in enumerate(records):
        record["i"] = index

    first_args["offset"] = 0
    first_args["limit"] = len(records)
    first_args["out"] = str(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"args": first_args, "records": records}, f, indent=2)
    print(f"[merge] wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
