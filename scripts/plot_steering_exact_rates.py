import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_dump_csv(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["alpha"] = float(r["alpha"])
            r["score"] = float(r["score"])
            r["refuse"] = int(r["refuse"])
            rows.append(r)
    return rows


def group_rates(rows: List[dict], keys: Tuple[str, ...]) -> Dict[Tuple, Tuple[int, int]]:
    """
    Returns: key_tuple -> (refuse_count, total_count)
    """
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        k = tuple(r[x] for x in keys)
        counts[k][0] += r["refuse"]
        counts[k][1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def group_means(rows: List[dict], keys: Tuple[str, ...]) -> Dict[Tuple, Tuple[float, int]]:
    """
    Returns: key_tuple -> (sum_score, n)
    """
    acc = defaultdict(lambda: [0.0, 0])
    for r in rows:
        k = tuple(r[x] for x in keys)
        acc[k][0] += r["score"]
        acc[k][1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


def plot_overall_rates(rows: List[dict], outpath: str) -> None:
    # model x split x alpha => rate
    rates = group_rates(rows, ("model", "split", "alpha"))
    alphas = sorted({r["alpha"] for r in rows})
    models = sorted({r["model"] for r in rows})
    splits = sorted({r["split"] for r in rows})

    plt.figure(figsize=(10, 6))
    for model in models:
        for split in splits:
            ys = []
            for a in alphas:
                rc, n = rates.get((model, split, a), (0, 0))
                ys.append((rc / n) if n > 0 else 0.0)
            plt.plot(alphas, ys, marker="o", label=f"{model}-{split}")

    plt.ylim(-0.05, 1.05)
    plt.xlabel("alpha")
    plt.ylabel("exact refusal rate = (#refuse)/(#total)")
    plt.title("Exact refusal-rate vs alpha (by split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_overall_means(rows: List[dict], outpath: str) -> None:
    means = group_means(rows, ("model", "split", "alpha"))
    alphas = sorted({r["alpha"] for r in rows})
    models = sorted({r["model"] for r in rows})
    splits = sorted({r["split"] for r in rows})

    plt.figure(figsize=(10, 6))
    for model in models:
        for split in splits:
            ys = []
            for a in alphas:
                ssum, n = means.get((model, split, a), (0.0, 0))
                ys.append((ssum / n) if n > 0 else 0.0)
            plt.plot(alphas, ys, marker="o", label=f"{model}-{split}")

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("alpha")
    plt.ylabel("mean refusal score")
    plt.title("Mean refusal-score vs alpha (by split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_edge_by_category(rows: List[dict], outpath: str) -> None:
    # only edge split
    edge = [r for r in rows if r["split"] == "edge"]
    rates = group_rates(edge, ("model", "category", "alpha"))
    alphas = sorted({r["alpha"] for r in edge})
    models = sorted({r["model"] for r in edge})
    cats = sorted({r["category"] for r in edge})

    # make one plot per model (less clutter)
    for model in models:
        plt.figure(figsize=(12, 7))
        for cat in cats:
            ys = []
            for a in alphas:
                rc, n = rates.get((model, cat, a), (0, 0))
                ys.append((rc / n) if n > 0 else 0.0)
            plt.plot(alphas, ys, marker="o", label=cat)

        plt.ylim(-0.05, 1.05)
        plt.xlabel("alpha")
        plt.ylabel("exact refusal rate")
        plt.title(f"Exact refusal-rate vs alpha — EDGE split by category ({model})")
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        base, ext = os.path.splitext(outpath)
        plt.savefig(f"{base}_{model}{ext}", dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_csv", required=True)
    ap.add_argument("--outdir", default="runs/plots_qwen25_exact")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = read_dump_csv(args.dump_csv)

    out_rates = os.path.join(args.outdir, "steering_exact_rates_overall.png")
    out_means = os.path.join(args.outdir, "steering_means_overall.png")
    out_edge_cat = os.path.join(args.outdir, "steering_exact_rates_edge_by_category.png")

    plot_overall_rates(rows, out_rates)
    plot_overall_means(rows, out_means)
    plot_edge_by_category(rows, out_edge_cat)

    print("[OK] wrote:", out_rates)
    print("[OK] wrote:", out_means)
    print("[OK] wrote:", out_edge_cat, "(and _base/_inst variants)")


if __name__ == "__main__":
    main()
