import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_score_hists(df, split, out_prefix):
    sub = df[df["split"] == split]
    plt.figure(figsize=(7,4))
    plt.hist(sub["s_base"], bins=25, alpha=0.6, label="base")
    plt.hist(sub["s_inst"], bins=25, alpha=0.6, label="instruct")
    plt.xlabel("refusal_score (logsumexp(refuse_tokens) - logsumexp(ok_tokens))")
    plt.ylabel("count")
    plt.title(f"Score distributions: {split}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_hist_{split}.png", dpi=200)
    plt.close()

def plot_scatter(df, out_prefix):
    plt.figure(figsize=(5.2,5))
    for split in sorted(df["split"].unique()):
        sub = df[df["split"] == split]
        plt.scatter(sub["s_base"], sub["s_inst"], s=12, alpha=0.7, label=split)
    plt.xlabel("s_base")
    plt.ylabel("s_inst")
    plt.title("Prompt-wise scores: base vs instruct")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_scatter.png", dpi=200)
    plt.close()

def plot_patch_sites(patch_json_path, split, out_prefix):
    j = json.load(open(patch_json_path, "r", encoding="utf-8"))
    sites = list(j["delta_score_mean"].keys())
    n_layers = j["n_layers"]

    mat = np.zeros((len(sites), n_layers))
    for i, site in enumerate(sites):
        for l in range(n_layers):
            mat[i, l] = j["delta_score_mean"][site][str(l)][split]

    plt.figure(figsize=(9,3.6))
    plt.imshow(mat, aspect="auto")
    plt.yticks(range(len(sites)), sites)
    plt.xticks(range(n_layers), range(n_layers))
    plt.xlabel("layer")
    plt.title(f"Δ refusal_score by site × layer ({split})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_heatmap_{split}.png", dpi=200)
    plt.close()

def plot_resid_curve(patch_json_path, out_prefix):
    j = json.load(open(patch_json_path, "r", encoding="utf-8"))
    n_layers = j["n_layers"]
    resid = j["delta_score_mean"]["hook_resid_pre"]
    layers = list(range(n_layers))
    harmful = [resid[str(l)]["harmful"] for l in layers]
    edge = [resid[str(l)]["edge"] for l in layers]

    plt.figure(figsize=(8,4))
    plt.plot(layers, harmful, label="harmful")
    plt.plot(layers, edge, label="edge")
    plt.axhline(0)
    plt.xlabel("layer")
    plt.ylabel("mean Δscore (patched - base)")
    plt.title("Activation patching profile (hook_resid_pre @ last position)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_resid_curve.png", dpi=200)
    plt.close()

def main():
    scores_csv = "runs/qwen25_scores.csv"
    patch_sites = "runs/patch_sites_qwen25.json"
    out_prefix = "plots/fig"

    df = pd.read_csv(scores_csv)
    plot_score_hists(df, "harmful", out_prefix)
    plot_score_hists(df, "edge", out_prefix)
    plot_scatter(df, out_prefix)

    plot_patch_sites(patch_sites, "harmful", out_prefix)
    plot_patch_sites(patch_sites, "edge", out_prefix)
    plot_resid_curve(patch_sites, out_prefix)

    print("Saved figures to plots/:")
    print(" - fig_hist_harmful.png / fig_hist_edge.png")
    print(" - fig_scatter.png")
    print(" - fig_heatmap_harmful.png / fig_heatmap_edge.png")
    print(" - fig_resid_curve.png")

if __name__ == "__main__":
    main()
