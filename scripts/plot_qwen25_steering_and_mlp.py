import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steering_json", default="runs/steering_balanced_qwen25.json")
    ap.add_argument("--mlp_lastk_json", default="runs/patch_lastK_mlp_out_qwen25.json")
    ap.add_argument("--outdir", default="runs/plots_qwen25")
    ap.add_argument("--threshold", type=float, default=0.0, help="refuse if score > threshold")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    steer = load_json(args.steering_json)
    mlp = load_json(args.mlp_lastk_json)

    # -----------------------
    # 1) steering mean curves
    # -----------------------
    alphas = [r["alpha"] for r in steer["results"]]
    base_edge = [r["base_edge_meanstd"][0] for r in steer["results"]]
    base_harm = [r["base_harm_meanstd"][0] for r in steer["results"]]
    inst_edge = [r["inst_edge_meanstd"][0] for r in steer["results"]]
    inst_harm = [r["inst_harm_meanstd"][0] for r in steer["results"]]

    plt.figure()
    plt.plot(alphas, base_edge, label="base edge (mean)")
    plt.plot(alphas, base_harm, label="base harmful (mean)")
    plt.plot(alphas, inst_edge, label="inst edge (mean)")
    plt.plot(alphas, inst_harm, label="inst harmful (mean)")
    plt.axhline(args.threshold, linestyle="--")
    plt.xlabel("alpha")
    plt.ylabel("mean refusal_score")
    plt.title("Steering curves (means)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "steering_means.png"), dpi=200)

    # ---------------------------------------
    # 2) approximate refusal-rate using Gauss
    # (rate = P(score > thr) under N(mean,std))
    # NOTE: this is an approximation since you didn't store per-prompt scores.
    # ---------------------------------------
    # We'll use normal approximation from mean/std in the json.
    def approx_rate(mean, std, thr):
        std = max(std, 1e-6)
        z = (thr - mean) / std
        # Phi(z) using erf
        return 0.5 * (1.0 - np.math.erf(z / np.sqrt(2.0)))

    base_edge_rate = [approx_rate(r["base_edge_meanstd"][0], r["base_edge_meanstd"][1], args.threshold) for r in steer["results"]]
    base_harm_rate = [approx_rate(r["base_harm_meanstd"][0], r["base_harm_meanstd"][1], args.threshold) for r in steer["results"]]
    inst_edge_rate = [approx_rate(r["inst_edge_meanstd"][0], r["inst_edge_meanstd"][1], args.threshold) for r in steer["results"]]
    inst_harm_rate = [approx_rate(r["inst_harm_meanstd"][0], r["inst_harm_meanstd"][1], args.threshold) for r in steer["results"]]

    plt.figure()
    plt.plot(alphas, base_edge_rate, label="base edge (approx rate)")
    plt.plot(alphas, base_harm_rate, label="base harmful (approx rate)")
    plt.plot(alphas, inst_edge_rate, label="inst edge (approx rate)")
    plt.plot(alphas, inst_harm_rate, label="inst harmful (approx rate)")
    plt.xlabel("alpha")
    plt.ylabel(f"P(score > {args.threshold})  (normal approx)")
    plt.title("Steering curves (approx refusal rate from mean/std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "steering_rates_approx.png"), dpi=200)

    # -----------------------------
    # 3) cumulative MLP patch curve
    # -----------------------------
    n_layers = mlp["n_layers"]
    Ks = list(range(1, n_layers + 1))
    d_harm = [mlp["delta_score_mean"][str(K)]["harmful"] for K in Ks]
    d_edge = [mlp["delta_score_mean"][str(K)]["edge"] for K in Ks]

    plt.figure()
    plt.plot(Ks, d_harm, label="harmful")
    plt.plot(Ks, d_edge, label="edge")
    plt.xlabel("K (patch last-K layers' hook_mlp_out)")
    plt.ylabel("mean Δscore (patched - base)")
    plt.title("Cumulative last-K MLP patching effect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "mlp_cumulative_delta.png"), dpi=200)

    # --------------------------------
    # 4) marginal contribution per K
    # layer_added = n_layers - K
    # marginal(K) = delta(K) - delta(K-1)
    # --------------------------------
    marg_harm = [d_harm[0]] + [d_harm[i] - d_harm[i - 1] for i in range(1, len(d_harm))]
    marg_edge = [d_edge[0]] + [d_edge[i] - d_edge[i - 1] for i in range(1, len(d_edge))]

    # Map each K to the layer index added when going from K-1 -> K
    # When K=1, layer_added = n_layers-1 (last layer)
    layer_added = [n_layers - K for K in Ks]

    plt.figure(figsize=(10, 4))
    plt.plot(layer_added, marg_harm, label="harmful marginal")
    plt.plot(layer_added, marg_edge, label="edge marginal")
    plt.xlabel("layer added (higher = later layer)")
    plt.ylabel("marginal Δscore")
    plt.title("Marginal contribution of each added layer (from cumulative MLP patching)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "mlp_marginal_by_layer.png"), dpi=200)

    print("Saved plots to:", args.outdir)


if __name__ == "__main__":
    main()
