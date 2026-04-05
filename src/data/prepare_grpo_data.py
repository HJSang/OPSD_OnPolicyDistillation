"""
Prepare data for GRPO training.

Splits DAPO-Math-17k-dedup 80/20 into train/val, and processes
MATH-500, AIME24, AIME25 as additional validation sets.

All outputs use the verl RL data format:
  - data_source: str
  - prompt: list[dict] (chat messages)
  - ability: str
  - reward_model: dict with style + ground_truth
  - extra_info: dict

Usage:
    python prepare_grpo_data.py \
        --data-dir /path/to/data \
        --output-dir /path/to/output \
        --train-ratio 0.8
"""

import argparse
import json
import os
import re
from pathlib import Path

import datasets
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Boxed instruction (aligned with Qwen3's default \boxed{} output format)
BOXED_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

# DAPO instruction (to be stripped from existing DAPO prompts)
DAPO_INSTRUCTION_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_INSTRUCTION_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def extract_boxed_answer(text):
    """Extract answer from \\boxed{} format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1) if match else text


def process_dapo(data_path, train_ratio=0.8, seed=42):
    """Process DAPO-Math-17k-dedup with train/val split.

    Returns:
        (train_dataset, val_dataset)
    """
    print(f"Loading DAPO-Math-17k-dedup from {data_path}...")
    df = pd.read_parquet(data_path)
    dataset = datasets.Dataset.from_pandas(df)

    def strip_dapo_instruction(text):
        """Remove DAPO instruction prefix/suffix, add boxed instruction."""
        # Strip prefix
        if text.startswith(DAPO_INSTRUCTION_PREFIX):
            text = text[len(DAPO_INSTRUCTION_PREFIX):]
        # Strip suffix
        if text.endswith(DAPO_INSTRUCTION_SUFFIX):
            text = text[: -len(DAPO_INSTRUCTION_SUFFIX)]
        # Add boxed instruction
        return text.rstrip() + "\n\n" + BOXED_INSTRUCTION

    def process_fn(example, idx):
        if "prompt" in example and isinstance(example["prompt"], list):
            prompt = example["prompt"]
            # Strip DAPO instruction from user message content
            for msg in prompt:
                if msg["role"] == "user":
                    msg["content"] = strip_dapo_instruction(msg["content"])
        else:
            prompt_content = example.get("prompt", "")
            prompt_content = strip_dapo_instruction(prompt_content)
            prompt = [{"role": "user", "content": prompt_content}]

        return {
            "data_source": "math_dapo",
            "prompt": prompt,
            "ability": "math",
            "reward_model": example.get("reward_model", {}),
            "extra_info": {"index": idx, "data_source": "math_dapo"},
        }

    dataset = dataset.map(function=process_fn, with_indices=True)
    dataset = dataset.select_columns(
        ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    )

    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    split_idx = int(len(dataset) * train_ratio)
    train_dataset = dataset.select(range(split_idx))
    val_dataset = dataset.select(range(split_idx, len(dataset)))

    # Tag splits
    train_dataset = train_dataset.map(
        lambda ex: {
            "extra_info": {**ex["extra_info"], "split": "train"}
        }
    )
    val_dataset = val_dataset.map(
        lambda ex: {
            "extra_info": {**ex["extra_info"], "split": "val"}
        }
    )

    print(f"  DAPO total: {len(dataset)}, train: {len(train_dataset)}, val: {len(val_dataset)}")
    return train_dataset, val_dataset


def process_aime24(data_path):
    """Process AIME 2024 validation data."""
    print(f"Loading AIME 2024 from {data_path}...")
    df = pd.read_parquet(data_path)
    dataset = datasets.Dataset.from_pandas(df)

    def process_fn(example, idx):
        question = example["problem"].rstrip() + "\n\n" + BOXED_INSTRUCTION
        solution = extract_boxed_answer(example["solution"])
        return {
            "data_source": "aime24",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": "test", "index": idx},
        }

    dataset = dataset.map(function=process_fn, with_indices=True)
    dataset = dataset.select_columns(
        ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    )
    print(f"  AIME 2024: {len(dataset)} examples")
    return dataset


def process_aime25(data_path):
    """Process AIME 2025 validation data."""
    print(f"Loading AIME 2025 from {data_path}...")
    data_list = []
    with open(data_path, "r") as f:
        for line in f:
            data_list.append(json.loads(line))
    dataset = datasets.Dataset.from_list(data_list)

    def process_fn(example, idx):
        question = example["problem"].rstrip() + "\n\n" + BOXED_INSTRUCTION
        solution = str(example["answer"])
        return {
            "data_source": "aime25",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": "test", "index": idx},
        }

    dataset = dataset.map(function=process_fn, with_indices=True)
    dataset = dataset.select_columns(
        ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    )
    print(f"  AIME 2025: {len(dataset)} examples")
    return dataset


def process_math500(data_path):
    """Process MATH-500 validation data."""
    print(f"Loading MATH-500 from {data_path}...")
    data_list = []
    with open(data_path, "r") as f:
        for line in f:
            data_list.append(json.loads(line))
    dataset = datasets.Dataset.from_list(data_list)

    def process_fn(example, idx):
        question = example["problem"].rstrip() + "\n\n" + BOXED_INSTRUCTION
        solution = str(example["answer"])
        return {
            "data_source": "math",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": "test", "index": idx},
        }

    dataset = dataset.map(function=process_fn, with_indices=True)
    dataset = dataset.select_columns(
        ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    )
    print(f"  MATH-500: {len(dataset)} examples")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO training data")
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data"),
        help="Root directory containing all data folders",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "grpo_processed"),
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of DAPO to use for training (default: 0.8)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- DAPO 80/20 split ---
    print("\n" + "=" * 80)
    print(f"Processing DAPO-Math-17k-dedup ({args.train_ratio:.0%} train / {1 - args.train_ratio:.0%} val)")
    print("=" * 80)

    dapo_path = os.path.join(
        data_dir, "DAPO-Math-17k-dedup", "distinct-prompts-with-rewards.parquet"
    )
    train_dataset, val_dapo_dataset = process_dapo(
        dapo_path, train_ratio=args.train_ratio, seed=args.seed
    )

    train_path = os.path.join(output_dir, "train.parquet")
    train_dataset.to_parquet(train_path)
    print(f"Saved train -> {train_path}")

    val_dapo_path = os.path.join(output_dir, "val_dapo.parquet")
    val_dapo_dataset.to_parquet(val_dapo_path)
    print(f"Saved val_dapo -> {val_dapo_path}")

    # --- Validation benchmarks ---
    print("\n" + "=" * 80)
    print("Processing Validation Benchmarks")
    print("=" * 80)

    aime24_path = os.path.join(data_dir, "aime24", "test-00000-of-00001.parquet")
    aime24_dataset = process_aime24(aime24_path)
    aime24_out = os.path.join(output_dir, "val_aime24.parquet")
    aime24_dataset.to_parquet(aime24_out)

    aime25_path = os.path.join(data_dir, "aime25", "test.jsonl")
    aime25_dataset = process_aime25(aime25_path)
    aime25_out = os.path.join(output_dir, "val_aime25.parquet")
    aime25_dataset.to_parquet(aime25_out)

    math500_path = os.path.join(data_dir, "MATH-500", "test.jsonl")
    math500_dataset = process_math500(math500_path)
    math500_out = os.path.join(output_dir, "val_math500.parquet")
    math500_dataset.to_parquet(math500_out)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Train (DAPO {args.train_ratio:.0%}): {len(train_dataset)}")
    print(f"Val DAPO ({1 - args.train_ratio:.0%}):   {len(val_dapo_dataset)}")
    print(f"Val AIME24:          {len(aime24_dataset)}")
    print(f"Val AIME25:          {len(aime25_dataset)}")
    print(f"Val MATH-500:        {len(math500_dataset)}")
    print(f"\nAll saved to: {output_dir}")


if __name__ == "__main__":
    main()
