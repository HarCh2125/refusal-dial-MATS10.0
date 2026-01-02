import json
import matplotlib.pyplot as plt

# Load the JSON file
with open("runs/patch_sites_qwen25.json", "r") as f:
    data = json.load(f)

hooks = ["hook_resid_pre", "hook_attn_out", "hook_mlp_out", "hook_resid_post"]
colors = {
    "hook_resid_pre": "blue",
    "hook_attn_out": "green",
    "hook_mlp_out": "orange",
    "hook_resid_post": "red"
}

markers = {
    "harmful": "o",
    "edge": "s"
}

plt.figure(figsize=(14, 7))

for hook in hooks:
    layers = sorted(data["delta_score_mean"][hook].keys(), key=int)
    harmful_values = [data["delta_score_mean"][hook][layer]["harmful"] for layer in layers]
    edge_values = [data["delta_score_mean"][hook][layer]["edge"] for layer in layers]
    
    plt.plot(layers, harmful_values, marker=markers["harmful"], color=colors[hook],
             linestyle="-", label=f"{hook} - harmful")
    plt.plot(layers, edge_values, marker=markers["edge"], color=colors[hook],
             linestyle="--", label=f"{hook} - edge")

plt.xlabel("Layer")
plt.ylabel("Delta Score Mean")
plt.title("Delta Score Mean per Layer for Different Hooks")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/delta_score_mean_per_layer_qwen25.png')