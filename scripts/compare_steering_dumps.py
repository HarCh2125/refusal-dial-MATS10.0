import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_meta_path(dump_csv: str) -> Optional[str]:
    """
    Tries common patterns:
      - replace '_dump.csv' -> '_meta.json'
      - replace '.csv' -> '_meta.json'
    """
    if dump_csv.endswith("_dump.csv"):
        cand = dump_csv.replace("_dump.csv", "_meta.json")
        if os.path.exists(cand):
            return cand
    cand2 = dump_csv.replace(".csv", "_meta.json")
    if os.path.exists(cand2):
        return cand2
    return None


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


def rate_table(rows: List[dict]) -> Dict[Tuple[str, str, float], Tuple[int, int]]:
    """
    (model, split, alpha) -> (refuse_count, total_count)
    """
    tab = defaultdict(lambda: [0, 0])
    for r in rows:
        k = (r["model"], r["split"], float(r["alpha"]))
        tab[k][0] += int(r["refuse"])
        tab[k][1] += 1
    return {k: (v[0], v[1]) for k, v in tab.items()}


def mean_table(rows: List[dict]) -> Dict[Tuple[str, str, float], Tuple[float, int]]:
    """
    (model, split, alpha) -> (sum_score, n)
    """
    tab = defaultdict(lambda: [0.0, 0])
    for r in rows:
        k = (r["model"], r["split"], float(r["alpha"]))
        tab[k][0] += float(r["score"])
        tab[k][1] += 1
    return {k: (v[0], v[1]) for k, v in tab.items()}


def pick_alpha_meeting_threshold(alphas: List[float], rates: Dict[float, float], target: float) -> Optional[float]:
    """
    Among alphas where rate <= target, pick the one with minimal |alpha|.
    Tie-break: alpha closer to 0 (same), then smaller alpha (deterministic).
    """
    candidates = [a for a in alphas if (a in rates and rates[a] <= target)]
    if not candidates:
        return None
    candidates.sort(key=lambda a: (abs(a), abs(a), a))
    return candidates[0]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_overlay(
    runs: List[dict],
    outpath: str,
    focus_model: str,
    split: str,
    ylabel: str,
    title: str,
    use_gap: bool = False,
) -> None:
    """
    runs[i] expects:
      run["label"], run["alphas"], run["rates"][model][split][alpha]
    if use_gap: plot (harmful-edge) gap for focus_model
    """
    plt.figure(figsize=(10, 6))

    for run in runs:
        label = run["label"]
        alphas = run["alphas"]

        if use_gap:
            ys = []
            for a in alphas:
                hr = run["rates"][focus_model]["harmful"].get(a, None)
                er = run["rates"][focus_model]["edge"].get(a, None)
                if hr is None or er is None:
                    ys.append(None)
                else:
                    ys.append(hr - er)
            # replace None with NaN-ish (matplotlib will break lines); simplest: skip if missing
            xs2, ys2 = [], []
            for x, y in zip(alphas, ys):
                if y is not None:
                    xs2.append(x)
                    ys2.append(y)
            plt.plot(xs2, ys2, marker="o", label=label)
        else:
            ys = []
            for a in alphas:
                v = run["rates"][focus_model][split].get(a, None)
                ys.append(v if v is not None else float("nan"))
            plt.plot(alphas, ys, marker="o", label=label)

    if not use_gap:
        plt.ylim(-0.05, 1.05)

    plt.xlabel("alpha")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def write_table_csv(rows: List[dict], out_csv: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_table_md(rows: List[dict], out_md: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True, help="One or more *_dump.csv files.")
    ap.add_argument("--metas", nargs="*", default=None, help="Optional meta JSONs (same order as dumps).")
    ap.add_argument("--outdir", default="runs/plots_qwen25_exact/compare")

    ap.add_argument("--focus_model", default="inst", choices=["inst", "base"])
    ap.add_argument("--edge_target", type=float, default=0.10)
    ap.add_argument("--harm_target", type=float, default=0.50)

    ap.add_argument("--also_plot_base", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Resolve meta paths
    meta_paths = []
    if args.metas and len(args.metas) > 0:
        if len(args.metas) != len(args.dumps):
            raise ValueError("If you pass --metas, provide exactly as many as --dumps (same order).")
        meta_paths = args.metas
    else:
        for d in args.dumps:
            mp = infer_meta_path(d)
            meta_paths.append(mp)

    runs = []
    for dump_csv, meta_path in zip(args.dumps, meta_paths):
        rows = read_dump_csv(dump_csv)

        meta = None
        if meta_path is not None and os.path.exists(meta_path):
            meta = read_json(meta_path)

        # label
        if meta:
            layer = meta.get("layer", "NA")
            site = meta.get("site", "NA")
            label = f"layer{layer}_{site}"
        else:
            label = os.path.basename(dump_csv).replace(".csv", "")

        alphas = sorted({float(r["alpha"]) for r in rows})
        rt = rate_table(rows)
        mt = mean_table(rows)

        # build easy dicts
        rates = {"base": {"edge": {}, "harmful": {}}, "inst": {"edge": {}, "harmful": {}}}
        means = {"base": {"edge": {}, "harmful": {}}, "inst": {"edge": {}, "harmful": {}}}

        for model in ["base", "inst"]:
            for split in ["edge", "harmful"]:
                for a in alphas:
                    rc, n = rt.get((model, split, a), (0, 0))
                    rates[model][split][a] = (rc / n) if n > 0 else None
                    ssum, n2 = mt.get((model, split, a), (0.0, 0))
                    means[model][split][a] = (ssum / n2) if n2 > 0 else None

        runs.append({
            "dump_csv": dump_csv,
            "meta_path": meta_path,
            "label": label,
            "alphas": alphas,
            "rates": rates,
            "means": means,
            "meta": meta,
        })

    # ----- Plots (focus model) -----
    out_edge = os.path.join(args.outdir, f"overlay_edge_rate_{args.focus_model}.png")
    out_harm = os.path.join(args.outdir, f"overlay_harmful_rate_{args.focus_model}.png")
    out_gap  = os.path.join(args.outdir, f"overlay_gap_hminus_e_{args.focus_model}.png")

    plot_overlay(
        runs, out_edge, args.focus_model, "edge",
        ylabel="exact refusal rate", title=f"EDGE refusal rate vs alpha ({args.focus_model})"
    )
    plot_overlay(
        runs, out_harm, args.focus_model, "harmful",
        ylabel="exact refusal rate", title=f"HARMFUL refusal rate vs alpha ({args.focus_model})"
    )
    plot_overlay(
        runs, out_gap, args.focus_model, "edge",
        ylabel="gap = harmful_rate - edge_rate",
        title=f"Selectivity gap vs alpha ({args.focus_model})",
        use_gap=True
    )

    # optional base overlays too
    if args.also_plot_base:
        out_edge_b = os.path.join(args.outdir, "overlay_edge_rate_base.png")
        out_harm_b = os.path.join(args.outdir, "overlay_harmful_rate_base.png")
        out_gap_b  = os.path.join(args.outdir, "overlay_gap_hminus_e_base.png")
        plot_overlay(runs, out_edge_b, "base", "edge", "exact refusal rate", "EDGE refusal rate vs alpha (base)")
        plot_overlay(runs, out_harm_b, "base", "harmful", "exact refusal rate", "HARMFUL refusal rate vs alpha (base)")
        plot_overlay(runs, out_gap_b, "base", "edge", "gap = harmful-edge", "Selectivity gap vs alpha (base)", use_gap=True)

    # ----- Summary table -----
    summary_rows = []
    for run in runs:
        label = run["label"]
        alphas = run["alphas"]
        for model in ["inst", "base"]:
            edge_rates = run["rates"][model]["edge"]
            harm_rates = run["rates"][model]["harmful"]

            # alpha for thresholds
            alpha_edge10 = pick_alpha_meeting_threshold(alphas, edge_rates, args.edge_target)
            alpha_harm50 = pick_alpha_meeting_threshold(alphas, harm_rates, args.harm_target)

            # gap at alpha=0
            gap0 = None
            if 0.0 in alphas and edge_rates.get(0.0) is not None and harm_rates.get(0.0) is not None:
                gap0 = harm_rates[0.0] - edge_rates[0.0]

            # best alpha by max gap (harm-edge)
            best_alpha = None
            best_gap = None
            best_edge = None
            best_harm = None
            for a in alphas:
                er = edge_rates.get(a, None)
                hr = harm_rates.get(a, None)
                if er is None or hr is None:
                    continue
                g = hr - er
                if (best_gap is None) or (g > best_gap) or (g == best_gap and abs(a) < abs(best_alpha)):
                    best_gap = g
                    best_alpha = a
                    best_edge = er
                    best_harm = hr

            def fmt(x):
                if x is None:
                    return ""
                if isinstance(x, float):
                    return f"{x:.4f}"
                return str(x)

            row = {
                "run": label,
                "model": model,
                "gap_at_alpha0": fmt(gap0),

                "alpha_edge<=target": fmt(alpha_edge10),
                "edge_rate@alpha_edge": fmt(edge_rates.get(alpha_edge10)) if alpha_edge10 is not None else "",
                "harm_rate@alpha_edge": fmt(harm_rates.get(alpha_edge10)) if alpha_edge10 is not None else "",
                "edge_target": fmt(args.edge_target),

                "alpha_harm<=target": fmt(alpha_harm50),
                "harm_rate@alpha_harm": fmt(harm_rates.get(alpha_harm50)) if alpha_harm50 is not None else "",
                "edge_rate@alpha_harm": fmt(edge_rates.get(alpha_harm50)) if alpha_harm50 is not None else "",
                "harm_target": fmt(args.harm_target),

                "best_alpha_by_max_gap": fmt(best_alpha),
                "best_gap": fmt(best_gap),
                "best_edge_rate": fmt(best_edge),
                "best_harm_rate": fmt(best_harm),
            }
            summary_rows.append(row)

    out_csv = os.path.join(args.outdir, "summary_thresholds_and_gap.csv")
    out_md  = os.path.join(args.outdir, "summary_thresholds_and_gap.md")
    write_table_csv(summary_rows, out_csv)
    write_table_md(summary_rows, out_md)

    print("[OK] wrote plots:")
    print(" -", out_edge)
    print(" -", out_harm)
    print(" -", out_gap)
    if args.also_plot_base:
        print(" - (also base overlays)")

    print("[OK] wrote tables:")
    print(" -", out_csv)
    print(" -", out_md)


if __name__ == "__main__":
    main()
