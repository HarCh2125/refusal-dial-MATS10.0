#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _alpha_key(d: Dict[str, Any], alpha: float) -> str:
    """
    Robust lookup for alpha keys like "-10.0" vs "-10".
    """
    candidates = [str(alpha), f"{alpha:.1f}", f"{alpha:.6g}"]
    for k in candidates:
        if k in d:
            return k
    # Last resort: try matching numerically
    for k in d.keys():
        try:
            if float(k) == float(alpha):
                return k
        except Exception:
            pass
    raise KeyError(f"Could not find alpha={alpha} in keys={list(d.keys())[:10]}...")


def pareto_front(points: pd.DataFrame, edge_col="edge_mean", harm_col="harm_mean") -> pd.DataFrame:
    """
    Pareto-optimal points for: minimize edge, maximize harmful.
    Sorted by edge ascending, keep points with strictly increasing harmful.
    """
    df = points.sort_values(edge_col, ascending=True).reset_index(drop=True)
    keep = []
    best_h = -1.0
    for i, row in df.iterrows():
        h = row[harm_col]
        if h > best_h + 1e-12:
            keep.append(True)
            best_h = h
        else:
            keep.append(False)
    df["pareto"] = keep
    return df


def aggregate_holdout(holdout_paths: List[str], model: str) -> pd.DataFrame:
    rows = []
    for fp in holdout_paths:
        j = _load_json(fp)
        alphas = [float(a) for a in j["alphas"]]
        rs = j["results_summary"][model]
        for a in alphas:
            ak = _alpha_key(rs["harmful"], a)
            edge = rs["edge_holdout"][ak]["refusal_rate"]
            harm = rs["harmful"][ak]["refusal_rate"]
            rows.append({"seed_file": os.path.basename(fp), "alpha": float(a), "edge": edge, "harm": harm, "gap": harm - edge})
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("alpha", as_index=False)
          .agg(edge_mean=("edge", "mean"), edge_std=("edge", "std"),
               harm_mean=("harm", "mean"), harm_std=("harm", "std"),
               gap_mean=("gap", "mean"), gap_std=("gap", "std"))
          .sort_values("alpha")
          .reset_index(drop=True)
    )
    return agg


def aggregate_random(random_json_path: str, model: str) -> pd.DataFrame:
    j = _load_json(random_json_path)
    alphas = [float(a) for a in j["alphas"]]
    rows = []
    for seed, d in j["results_summary"].items():
        rs = d[model]
        for a in alphas:
            ak = _alpha_key(rs["harmful"], a)
            edge = rs["edge"][ak]["refusal_rate"]
            harm = rs["harmful"][ak]["refusal_rate"]
            rows.append({"seed": str(seed), "alpha": float(a), "edge": edge, "harm": harm, "gap": harm - edge})
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("alpha", as_index=False)
          .agg(edge_mean=("edge", "mean"), edge_std=("edge", "std"),
               harm_mean=("harm", "mean"), harm_std=("harm", "std"),
               gap_mean=("gap", "mean"), gap_std=("gap", "std"))
          .sort_values("alpha")
          .reset_index(drop=True)
    )
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout_glob", required=True, help="e.g. runs/steering_holdout_layer20_hook_resid_pre_seed*.json")
    ap.add_argument("--random_json", required=True, help="e.g. runs/steering_random_layer20_hook_resid_pre_seeds0-19.json")
    ap.add_argument("--out_prefix", required=True, help="e.g. runs/controls_layer20_hook_resid_pre")
    ap.add_argument("--models", nargs="+", default=["inst"], choices=["inst", "base"])
    args = ap.parse_args()

    holdout_paths = sorted(glob.glob(args.holdout_glob))
    if len(holdout_paths) == 0:
        raise FileNotFoundError(f"No files matched holdout_glob: {args.holdout_glob}")

    for model in args.models:
        hold = aggregate_holdout(holdout_paths, model=model)
        rnd = aggregate_random(args.random_json, model=model)

        # Save CSV
        out_csv = f"{args.out_prefix}_{model}_controls_pareto.csv"
        merged = hold.merge(rnd, on="alpha", suffixes=("_holdout", "_random"))
        merged.to_csv(out_csv, index=False)

        # Best alpha by max (mean) gap
        best_idx = int(np.argmax(hold["gap_mean"].values))
        best_alpha = float(hold.loc[best_idx, "alpha"])
        best_gap = float(hold.loc[best_idx, "gap_mean"])

        # --- Plot 1: gap vs alpha (holdout vs random)
        plt.figure(figsize=(9, 5))
        plt.errorbar(hold["alpha"], hold["gap_mean"], yerr=hold["gap_std"], marker="o", capsize=3, label="holdout direction")
        plt.errorbar(rnd["alpha"],  rnd["gap_mean"],  yerr=rnd["gap_std"],  marker="o", capsize=3, label="random direction")
        plt.axvline(best_alpha, linestyle="--", linewidth=1)
        plt.title(f"Selectivity gap vs alpha ({model}) | best holdout alpha={best_alpha:g}, gap={best_gap:.3f}")
        plt.xlabel("alpha")
        plt.ylabel("gap = harmful_refusal_rate - edge_holdout_refusal_rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_{model}_gap_vs_alpha.png", dpi=200)
        plt.close()

        # --- Plot 2: Pareto (edge vs harmful)
        hold_pf = pareto_front(hold, edge_col="edge_mean", harm_col="harm_mean")

        plt.figure(figsize=(6.8, 6.2))
        # connecting curve (alpha-sorted)
        plt.errorbar(hold["edge_mean"], hold["harm_mean"], xerr=hold["edge_std"], yerr=hold["harm_std"],
                     marker="o", capsize=3, linestyle="-", label="holdout direction")
        plt.errorbar(rnd["edge_mean"], rnd["harm_mean"], xerr=rnd["edge_std"], yerr=rnd["harm_std"],
                     marker="o", capsize=3, linestyle="--", label="random direction")

        # highlight pareto points
        pareto_pts = hold_pf[hold_pf["pareto"]]
        plt.scatter(pareto_pts["edge_mean"], pareto_pts["harm_mean"], s=80, marker="D", label="holdout Pareto points")

        # annotate alphas lightly
        for _, r in hold.iterrows():
            plt.annotate(f"{r['alpha']:g}", (r["edge_mean"], r["harm_mean"]), fontsize=8, xytext=(4, 3), textcoords="offset points")

        plt.title(f"Pareto tradeoff ({model})  |  x=edge_holdout refusal, y=harmful refusal")
        plt.xlabel("edge_holdout refusal rate")
        plt.ylabel("harmful refusal rate")
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.02)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_{model}_pareto.png", dpi=220)
        plt.close()

        print(f"[OK] Wrote:\n  {out_csv}\n  {args.out_prefix}_{model}_gap_vs_alpha.png\n  {args.out_prefix}_{model}_pareto.png")

if __name__ == "__main__":
    main()
