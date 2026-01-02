#!/usr/bin/env python3
"""
Plot holdout-direction control vs random-direction baseline for steering runs.

Inputs:
  - Many holdout JSONs: runs/steering_holdout_layer20_hook_resid_pre_seed*.json
  - One random JSON:    runs/steering_random_layer20_hook_resid_pre_seeds0-19.json

Outputs:
  - out_dir/holdout_inst_rates.png
  - out_dir/holdout_base_rates.png
  - out_dir/random_mean_band_inst.png
  - out_dir/random_mean_band_base.png
  - out_dir/compare_inst_holdout_vs_random.png
  - out_dir/summary_controls.csv
"""
import argparse
import glob
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_alpha_key(d: Dict[str, Any], alpha: float) -> str:
    # JSON keys may be "0.0" or "0"
    k1 = str(float(alpha))
    if k1 in d:
        return k1
    k2 = str(alpha)
    if k2 in d:
        return k2
    # last resort: try formatting without trailing zeros
    k3 = f"{alpha:g}"
    if k3 in d:
        return k3
    raise KeyError(f"Could not find alpha={alpha} in keys like {list(d.keys())[:5]} ...")


def load_holdout_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    # expected: j["holdout"]["seed"], j["alphas"], j["results_summary"][model][split][alpha_key]["refusal_rate"]
    return j


def load_random_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    # expected: j["seeds"], j["alphas"], j["results_summary"][seed_str][model][split][alpha_key]["refusal_rate"]
    return j


def holdout_to_df(j: Dict[str, Any], path: str) -> pd.DataFrame:
    seed = j.get("holdout", {}).get("seed", None)
    alphas = [float(a) for a in j["alphas"]]
    rows = []
    for model in ["base", "inst"]:
        for split in ["edge_train", "edge_holdout", "harmful"]:
            d = j["results_summary"][model][split]
            for a in alphas:
                k = _get_alpha_key(d, a)
                rows.append(
                    {
                        "kind": "holdout",
                        "path": path,
                        "seed": seed,
                        "model": model,
                        "split": split,
                        "alpha": a,
                        "refusal_rate": float(d[k]["refusal_rate"]),
                    }
                )
    return pd.DataFrame(rows)


def random_to_df(j: Dict[str, Any], path: str) -> pd.DataFrame:
    alphas = [float(a) for a in j["alphas"]]
    rows = []
    for seed in j["seeds"]:
        rs = j["results_summary"][str(seed)]
        for model in ["base", "inst"]:
            for split in ["edge", "harmful"]:
                d = rs[model][split]
                for a in alphas:
                    k = _get_alpha_key(d, a)
                    rows.append(
                        {
                            "kind": "random",
                            "path": path,
                            "seed": int(seed),
                            "model": model,
                            "split": split,
                            "alpha": a,
                            "refusal_rate": float(d[k]["refusal_rate"]),
                        }
                    )
    return pd.DataFrame(rows)


def agg_mean_std(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(group_cols)["refusal_rate"].agg(["mean", "std", "count"]).reset_index()
    g.rename(columns={"mean": "rate_mean", "std": "rate_std", "count": "n"}, inplace=True)
    g["rate_std"] = g["rate_std"].fillna(0.0)
    return g


def plot_holdout_rates(df_hold: pd.DataFrame, out_path: str, model: str) -> None:
    d = df_hold[df_hold["model"] == model].copy()
    # aggregate across seeds if multiple
    a = agg_mean_std(d, ["split", "alpha"])
    plt.figure()
    for split in ["edge_train", "edge_holdout", "harmful"]:
        s = a[a["split"] == split].sort_values("alpha")
        plt.plot(s["alpha"], s["rate_mean"], marker="o", label=f"{split} (mean)")
        if s["n"].max() > 1:
            plt.fill_between(
                s["alpha"],
                np.clip(s["rate_mean"] - s["rate_std"], 0, 1),
                np.clip(s["rate_mean"] + s["rate_std"], 0, 1),
                alpha=0.2,
            )
    plt.ylim(-0.02, 1.02)
    plt.xlabel("alpha")
    plt.ylabel("exact refusal rate")
    plt.title(f"HOLDOUT control: refusal rate vs alpha ({model})")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_random_band(df_rnd: pd.DataFrame, out_path: str, model: str) -> None:
    d = df_rnd[df_rnd["model"] == model].copy()
    a = agg_mean_std(d, ["split", "alpha"]).sort_values("alpha")
    plt.figure()
    for split in ["edge", "harmful"]:
        s = a[a["split"] == split].sort_values("alpha")
        plt.plot(s["alpha"], s["rate_mean"], marker="o", label=f"{split} mean (n={int(s['n'].max())})")
        plt.fill_between(
            s["alpha"],
            np.clip(s["rate_mean"] - s["rate_std"], 0, 1),
            np.clip(s["rate_mean"] + s["rate_std"], 0, 1),
            alpha=0.2,
        )
    plt.ylim(-0.02, 1.02)
    plt.xlabel("alpha")
    plt.ylabel("exact refusal rate")
    plt.title(f"RANDOM baseline: refusal rate vs alpha ({model})")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_compare_inst(df_hold: pd.DataFrame, df_rnd: pd.DataFrame, out_path: str) -> None:
    # Compare inst edge_holdout (holdout direction) vs random inst edge band, plus harmful curves.
    h = df_hold[(df_hold["model"] == "inst")].copy()
    h_agg = agg_mean_std(h, ["split", "alpha"])

    r = df_rnd[(df_rnd["model"] == "inst")].copy()
    r_agg = agg_mean_std(r, ["split", "alpha"])

    # build series
    edge_hold = h_agg[h_agg["split"] == "edge_holdout"].sort_values("alpha")
    harm_hold = h_agg[h_agg["split"] == "harmful"].sort_values("alpha")

    edge_r = r_agg[r_agg["split"] == "edge"].sort_values("alpha")
    harm_r = r_agg[r_agg["split"] == "harmful"].sort_values("alpha")

    plt.figure()
    # holdout direction
    plt.plot(edge_hold["alpha"], edge_hold["rate_mean"], marker="o", label="holdout-direction: edge_holdout")
    if edge_hold["n"].max() > 1:
        plt.fill_between(
            edge_hold["alpha"],
            np.clip(edge_hold["rate_mean"] - edge_hold["rate_std"], 0, 1),
            np.clip(edge_hold["rate_mean"] + edge_hold["rate_std"], 0, 1),
            alpha=0.2,
        )
    plt.plot(harm_hold["alpha"], harm_hold["rate_mean"], marker="o", label="holdout-direction: harmful")
    if harm_hold["n"].max() > 1:
        plt.fill_between(
            harm_hold["alpha"],
            np.clip(harm_hold["rate_mean"] - harm_hold["rate_std"], 0, 1),
            np.clip(harm_hold["rate_mean"] + harm_hold["rate_std"], 0, 1),
            alpha=0.2,
        )

    # random band
    plt.plot(edge_r["alpha"], edge_r["rate_mean"], marker="o", linestyle="--", label="random: edge mean")
    plt.fill_between(
        edge_r["alpha"],
        np.clip(edge_r["rate_mean"] - edge_r["rate_std"], 0, 1),
        np.clip(edge_r["rate_mean"] + edge_r["rate_std"], 0, 1),
        alpha=0.15,
    )
    plt.plot(harm_r["alpha"], harm_r["rate_mean"], marker="o", linestyle="--", label="random: harmful mean")
    plt.fill_between(
        harm_r["alpha"],
        np.clip(harm_r["rate_mean"] - harm_r["rate_std"], 0, 1),
        np.clip(harm_r["rate_mean"] + harm_r["rate_std"], 0, 1),
        alpha=0.15,
    )

    plt.ylim(-0.02, 1.02)
    plt.xlabel("alpha")
    plt.ylabel("exact refusal rate")
    plt.title("INST: holdout-direction vs random-direction baseline")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_summary(df_hold: pd.DataFrame, df_rnd: pd.DataFrame) -> pd.DataFrame:
    # Key stats for INST and BASE:
    # - max|edge_train-edge_holdout| over alpha
    # - best alpha by max gap (harmful - edge_holdout for holdout; harmful-edge for random mean)
    out_rows = []

    for model in ["inst", "base"]:
        # holdout stats
        h = df_hold[df_hold["model"] == model].copy()
        ht = h[h["split"] == "edge_train"].set_index(["alpha", "seed"])["refusal_rate"].unstack("seed")
        hh = h[h["split"] == "edge_holdout"].set_index(["alpha", "seed"])["refusal_rate"].unstack("seed")

        # compare using per-seed means (then mean across seeds)
        ht_mean = ht.mean(axis=1)
        hh_mean = hh.mean(axis=1)
        gen_gap = (hh_mean - ht_mean).abs()
        gen_gap_mean = float(gen_gap.mean())
        gen_gap_max = float(gen_gap.max())

        harm = h[h["split"] == "harmful"].set_index(["alpha", "seed"])["refusal_rate"].unstack("seed").mean(axis=1)
        gap_hold = harm - hh_mean
        best_alpha_hold = float(gap_hold.idxmax())
        best_gap_hold = float(gap_hold.max())
        best_edge_hold = float(hh_mean.loc[best_alpha_hold])
        best_harm_hold = float(harm.loc[best_alpha_hold])

        # random stats (mean over seeds)
        r = df_rnd[df_rnd["model"] == model].copy()
        r_edge = r[r["split"] == "edge"].groupby("alpha")["refusal_rate"].mean()
        r_harm = r[r["split"] == "harmful"].groupby("alpha")["refusal_rate"].mean()
        gap_rnd = r_harm - r_edge
        best_alpha_rnd = float(gap_rnd.idxmax())
        best_gap_rnd = float(gap_rnd.max())
        best_edge_rnd = float(r_edge.loc[best_alpha_rnd])
        best_harm_rnd = float(r_harm.loc[best_alpha_rnd])

        out_rows.append(
            {
                "model": model,
                "holdout_gen_gap_mean": gen_gap_mean,
                "holdout_gen_gap_max": gen_gap_max,
                "holdout_best_alpha_by_max_gap": best_alpha_hold,
                "holdout_best_gap": best_gap_hold,
                "holdout_best_edge_holdout": best_edge_hold,
                "holdout_best_harmful": best_harm_hold,
                "random_best_alpha_by_max_gap": best_alpha_rnd,
                "random_best_gap": best_gap_rnd,
                "random_best_edge": best_edge_rnd,
                "random_best_harmful": best_harm_rnd,
            }
        )

    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout_json_glob", required=True, help='e.g. "runs/steering_holdout_layer20_hook_resid_pre_seed*.json"')
    ap.add_argument("--random_json", required=True, help="random-direction JSON (multi-seed)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    hold_paths = sorted(glob.glob(args.holdout_json_glob))
    if not hold_paths:
        raise FileNotFoundError(f"No files matched: {args.holdout_json_glob}")

    rnd = load_random_json(args.random_json)
    df_rnd = random_to_df(rnd, args.random_json)

    dfs = []
    for p in hold_paths:
        hj = load_holdout_json(p)
        dfs.append(holdout_to_df(hj, p))
    df_hold = pd.concat(dfs, ignore_index=True)

    os.makedirs(args.out_dir, exist_ok=True)

    # Plots
    plot_holdout_rates(df_hold, os.path.join(args.out_dir, "holdout_inst_rates.png"), model="inst")
    plot_holdout_rates(df_hold, os.path.join(args.out_dir, "holdout_base_rates.png"), model="base")

    plot_random_band(df_rnd, os.path.join(args.out_dir, "random_mean_band_inst.png"), model="inst")
    plot_random_band(df_rnd, os.path.join(args.out_dir, "random_mean_band_base.png"), model="base")

    plot_compare_inst(df_hold, df_rnd, os.path.join(args.out_dir, "compare_inst_holdout_vs_random.png"))

    # Summary CSV
    summary = compute_summary(df_hold, df_rnd)
    summary_path = os.path.join(args.out_dir, "summary_controls.csv")
    summary.to_csv(summary_path, index=False)
    print("Wrote:", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
