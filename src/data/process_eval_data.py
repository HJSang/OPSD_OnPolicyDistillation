"""
Prepare evaluation datasets for math benchmarks.

Generates validation parquet files for AIME24, AIME25, and MATH-500 using one
of three prompt styles:
  - boxed: add boxed-answer instruction
  - dapo: add DAPO-style answer instruction
  - none: leave the prompt unchanged
"""

import argparse
import json
import os
import re
from pathlib import Path

import datasets
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

BOXED_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
DAPO_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_REMINDER = '\n\nRemember to put your answer on its own line after "Answer:".'


def extract_boxed_answer(text: str) -> str:
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1) if match else text


def build_prompt(question: str, instruction_variant: str) -> str:
    question = question.rstrip()
    if instruction_variant == "boxed":
        return f"{question}\n\n{BOXED_INSTRUCTION}"
    if instruction_variant == "dapo":
        return f"{DAPO_INSTRUCTION}{question}{DAPO_REMINDER}"
    if instruction_variant == "none":
        return question
    raise ValueError(
        f"Unsupported instruction variant: {instruction_variant}. "
        "Expected one of: boxed, dapo, none."
    )


def process_aime24(data_path: str, instruction_variant: str):
    df = pd.read_parquet(data_path)
    dataset = datasets.Dataset.from_pandas(df)

    def process_fn(example, idx):
        return {
            "data_source": "aime24",
            "prompt": [
                {
                    "role": "user",
                    "content": build_prompt(example["problem"], instruction_variant),
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": extract_boxed_answer(example["solution"]),
            },
            "extra_info": {"split": "test", "index": idx},
        }

    dataset = dataset.map(function=process_fn, with_indices=True)
    return dataset.select_columns(
        ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    )


def process_jsonl_dataset(data_path: str):
    data_list = []
    with open(data_path, "r") as f:
        for line in f:
            data_list.append(json.loads(line))
    return datasets.Dataset.from_list(data_list)


def process_aime25(data_path: str, instruction_variant: str):
    dataset = process_jsonl_dataset(data_path)

    def process_fn(example, idx):
        return {
            "data_source": "aime25",
            "prompt": [
                {
                    "role": "user",
                    "content": build_prompt(example["problem"], instruction_variant),
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(example["answer"])},
            "extra_info": {"split": "test", "index": idx},
        }

    dataset = dataset.map(function=process_fn, with_indices=True)
    return dataset.select_columns(
        ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    )


def process_math500(data_path: str, instruction_variant: str):
    dataset = process_jsonl_dataset(data_path)

    def process_fn(example, idx):
        return {
            "data_source": "math",
            "prompt": [
                {
                    "role": "user",
                    "content": build_prompt(example["problem"], instruction_variant),
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(example["answer"])},
            "extra_info": {"split": "test", "index": idx},
        }

    dataset = dataset.map(function=process_fn, with_indices=True)
    return dataset.select_columns(
        ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare math evaluation data")
    parser.add_argument(
        "--data_dir",
        default=str(REPO_ROOT / "data"),
        help="Root directory containing the raw benchmark data",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write processed validation parquet files",
    )
    parser.add_argument(
        "--instruction_variant",
        choices=["boxed", "dapo", "none"],
        default="boxed",
        help="Prompt instruction style to apply",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    aime24_dataset = process_aime24(
        os.path.join(args.data_dir, "AIME_2024", "aime_2024_problems.parquet"),
        args.instruction_variant,
    )
    aime24_dataset.to_parquet(os.path.join(args.output_dir, "val_aime24.parquet"))

    aime25_dataset = process_aime25(
        os.path.join(args.data_dir, "AIME_2025", "train.jsonl"),
        args.instruction_variant,
    )
    aime25_dataset.to_parquet(os.path.join(args.output_dir, "val_aime25.parquet"))

    math500_dataset = process_math500(
        os.path.join(args.data_dir, "MATH-500", "test.jsonl"),
        args.instruction_variant,
    )
    math500_dataset.to_parquet(os.path.join(args.output_dir, "val_math500.parquet"))


if __name__ == "__main__":
    main()
