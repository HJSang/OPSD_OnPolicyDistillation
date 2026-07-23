#!/usr/bin/env python3
"""
Plot compression-ratio vs accuracy curves for every method on LongMemEval.

Reads result JSONs (two schemas):
  - run_validation.py outputs: records have '<cond>_ok', '<cond>_tokens',
    'full_tokens' (conditions raw/summary/dsocr).
  - eval_softtoken.py outputs: records have 'ok', 'tokens', 'soft_tokens'.

Each (file, condition) becomes one point (mean compression ratio, accuracy).
Points are grouped into method curves and plotted.

Usage:
    python plot_pareto.py
"""
import argparse
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))


def point_from_runval(path, cond):
    d = json.load(open(path))
    recs = d["records"]
    oks, ratios = [], []
    for r in recs:
        okk = f"{cond}_ok"
        if okk not in r:
            return None
        oks.append(1.0 if r[okk] else 0.0)
        tok = r.get(f"{cond}_tokens")
        full = r.get("full_tokens")
        if cond == "raw":
            ratios.append(1.0)
        elif tok and full:
            ratios.append(full / max(1, tok))
    if not oks:
        return None
    return (sum(ratios) / len(ratios), sum(oks) / len(oks), len(oks))


def point_from_softtoken(path):
    d = json.load(open(path))
    recs = d["records"]
    oks = [1.0 if r["ok"] else 0.0 for r in recs]
    ratios = [r["tokens"] / max(1, r["soft_tokens"]) for r in recs]
    return (sum(ratios) / len(ratios), sum(oks) / len(oks), len(oks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.environ.get(
        "VTC_RESULTS_DIR", os.path.join(ROOT, "results")))
    ap.add_argument("--out", default=os.path.join(ROOT, "pareto_longmemeval.png"))
    ap.add_argument("--glob", default="results_*.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, args.glob)))
    # method curves: name -> list of (ratio, acc, n, label)
    curves = defaultdict(list)

    for path in files:
        base = os.path.basename(path)
        # soft-token files
        if "softtoken" in base:
            pt = point_from_softtoken(path)
            if "simple" in base:
                curves["soft-token simple"].append(pt)
            elif "full_u1b" in base:
                curves["soft-token full (user-lossless)"].append(pt)
            elif "full_u1" in base:
                curves["soft-token full_u1 (lossy)"].append(pt)
            elif "full" in base:
                curves["soft-token full (u4/a16)"].append(pt)
            continue
        # run_validation files: detect conditions present
        d = json.load(open(path))
        r0 = d["records"][0]
        for cond in ("raw", "summary", "dsocr"):
            if f"{cond}_ok" in r0:
                pt = point_from_runval(path, cond)
                if pt is None:
                    continue
                if cond == "raw":
                    curves["raw (upper bound)"].append(pt)
                elif cond == "summary":
                    curves["text summary"].append(pt)
                elif cond == "dsocr":
                    label = ("DeepSeek-OCR full" if "full" in base
                             else "DeepSeek-OCR")
                    curves[label].append(pt)

    # dedupe + sort each curve by compression ratio
    for k in curves:
        pts = sorted(set(curves[k]), key=lambda p: p[0])
        curves[k] = pts

    # ---- print a table ----
    print(f"\n{'method':<32}{'ratio':>8}{'acc':>7}{'n':>5}")
    for name in curves:
        for (ratio, acc, n) in curves[name]:
            print(f"{name:<32}{ratio:>7.2f}x{acc:>7.3f}{n:>5}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable ({e}); wrote table only.")
        return

    plt.figure(figsize=(9, 6))
    markers = {"DeepSeek-OCR": "o-", "DeepSeek-OCR full": "s-",
               "soft-token simple": "^-", "soft-token full (u4/a16)": "v-",
               "soft-token full (user-lossless)": "D-",
               "text summary": "x--", "raw (upper bound)": "*"}
    for name, pts in sorted(curves.items()):
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        style = markers.get(name, "o-")
        if name.startswith("raw"):
            plt.axhline(ys[0], ls=":", color="gray", alpha=0.7,
                        label=f"{name} = {ys[0]:.2f}")
        else:
            plt.plot(xs, ys, style, label=name, markersize=8, linewidth=1.8)

    plt.xscale("log")
    plt.xlabel("compression ratio (x, log scale)")
    plt.ylabel("memory-QA accuracy (overall)")
    plt.title("LongMemEval: accuracy vs compression ratio (16 items)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
