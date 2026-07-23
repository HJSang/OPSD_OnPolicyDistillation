#!/usr/bin/env python3
"""Summarize one-seed Qwen2.5 base/Instruct reproductions of Tables 2 and 3."""

import argparse
import json
from pathlib import Path


TABLE2_METHODS = [
    ("SoftMem a8", "results_softtoken_u1_a8_pred_official_judge.json",
     "tokens", "soft_tokens"),
    ("SoftMem a16", "results_softtoken_u1_a16_pred_official_judge.json",
     "tokens", "soft_tokens"),
    ("SoftMem a32", "results_softtoken_u1_a32_pred_official_judge.json",
     "tokens", "soft_tokens"),
    ("DeepSeek-OCR b1024",
     "results_longmemeval_dsocr_b1024_dsocr_pred_official_judge.json",
     "full_tokens", "dsocr_tokens"),
    ("DeepSeek-OCR b640",
     "results_longmemeval_dsocr_b640_dsocr_pred_official_judge.json",
     "full_tokens", "dsocr_tokens"),
    ("DeepSeek-OCR b512",
     "results_longmemeval_dsocr_b512_dsocr_pred_official_judge.json",
     "full_tokens", "dsocr_tokens"),
    ("Text summary",
     "results_longmemeval_summary_summary_pred_official_judge.json",
     "full_tokens", "summary_tokens"),
]

PAPER_TABLE2 = {
    "SoftMem a8": 0.476,
    "SoftMem a16": 0.478,
    "SoftMem a32": 0.504,
    "DeepSeek-OCR b1024": 0.426,
    "DeepSeek-OCR b640": 0.390,
    "DeepSeek-OCR b512": 0.312,
    "Text summary": 0.184,
}

PAPER_TABLE3 = {
    "overall": 43.80,
    "narrativeqa": 29.29,
    "qasper": 44.14,
    "multifieldqa_en": 52.32,
    "hotpotqa": 58.40,
    "2wikimqa": 47.80,
    "musique": 30.85,
}


def load(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def mean_ratio(records, numerator, denominator):
    ratios = [
        row[numerator] / max(1, row[denominator])
        for row in records
    ]
    return sum(ratios) / len(ratios)


def summarize_variant(results_dir, label):
    table2 = {}
    for method, filename, numerator, denominator in TABLE2_METHODS:
        payload = load(results_dir / label / filename)
        table2[method] = {
            "compression": mean_ratio(
                payload["records"], numerator, denominator),
            "overall": payload["official_metrics"]["overall"],
            "user_fact": payload["official_metrics"]["by_category"]
            ["single-session-user"]["accuracy"],
        }

    table3_path = results_dir / label / f"result_table3_raw_{label}.json"
    subsets = load(table3_path)
    table3 = {
        "overall": sum(subsets.values()) / len(subsets),
        "subsets": subsets,
    }
    return {"table2": table2, "table3": table3}


def render_markdown(summary):
    lines = [
        "# Qwen2.5 Tables 2 and 3: one-seed reproduction",
        "",
        "Both variants use their tokenizer-provided chat template. SoftMem "
        "results use training seed 0.",
        "",
        "## Table 2: LongMemEval",
        "",
        "| Reader | Method | Compression | Overall | User-fact |",
        "|---|---|---:|---:|---:|",
    ]
    for label in ("base", "instruct"):
        for method, metrics in summary[label]["table2"].items():
            lines.append(
                f"| {label} | {method} | {metrics['compression']:.4f}x | "
                f"{metrics['overall']:.4f} | {metrics['user_fact']:.4f} |"
            )

    lines.extend([
        "",
        "### Instruct comparison with the paper",
        "",
        "| Method | Paper | Reproduced | Delta |",
        "|---|---:|---:|---:|",
    ])
    for method, paper in PAPER_TABLE2.items():
        reproduced = summary["instruct"]["table2"][method]["overall"]
        lines.append(
            f"| {method} | {paper:.3f} | {reproduced:.3f} | "
            f"{reproduced - paper:+.3f} |"
        )

    lines.extend([
        "",
        "## Table 3: raw LongBench-QA",
        "",
        "| Reader | Overall | narrativeqa | qasper | multifieldqa_en | "
        "hotpotqa | 2wikimqa | musique |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label in ("base", "instruct"):
        table3 = summary[label]["table3"]
        subsets = table3["subsets"]
        lines.append(
            f"| {label} | {table3['overall']:.2f} | "
            f"{subsets['narrativeqa']:.2f} | {subsets['qasper']:.2f} | "
            f"{subsets['multifieldqa_en']:.2f} | {subsets['hotpotqa']:.2f} | "
            f"{subsets['2wikimqa']:.2f} | {subsets['musique']:.2f} |"
        )

    lines.extend([
        "",
        "### Instruct comparison with the paper",
        "",
        "| Metric | Paper | Reproduced | Delta |",
        "|---|---:|---:|---:|",
    ])
    reproduced_table3 = summary["instruct"]["table3"]
    for metric, paper in PAPER_TABLE3.items():
        reproduced = (
            reproduced_table3["overall"]
            if metric == "overall"
            else reproduced_table3["subsets"][metric]
        )
        lines.append(
            f"| {metric} | {paper:.2f} | {reproduced:.2f} | "
            f"{reproduced - paper:+.2f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    args = parser.parse_args()

    results_dir = args.runs_dir / "results"
    summary = {
        label: summarize_variant(results_dir, label)
        for label in ("base", "instruct")
    }
    with (args.runs_dir / "comparison-summary.json").open(
            "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    markdown = render_markdown(summary)
    (args.runs_dir / "comparison-summary.md").write_text(
        markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
