import json
import torch
from transformer_lens import HookedTransformer
from experiments.utils import load_prompts, is_refusal

@torch.no_grad()
def compute_direction(model, prompts, layer: int, n_per_split: int = 50):
    # Use the model’s cached resid_pre on refused vs complied prompts
    refused, complied = [], []
    for ex in prompts:
        toks = model.to_tokens(ex["prompt"])
        logits, cache = model.run_with_cache(toks, names_filter=lambda n: f"blocks.{layer}.hook_resid_pre" in n)
        resid = cache[f"blocks.{layer}.hook_resid_pre"][0, -1]  # last token position
        # rough label via greedy generation (fast enough at small N)
        out = model.generate(toks, max_new_tokens=60, do_sample=False)
        txt = model.to_string(out[0])
        if is_refusal(txt) and len(refused) < n_per_split:
            refused.append(resid)
        elif (not is_refusal(txt)) and len(complied) < n_per_split:
            complied.append(resid)
        if len(refused) >= n_per_split and len(complied) >= n_per_split:
            break

    R = torch.stack(refused).mean(0)
    C = torch.stack(complied).mean(0)
    d = (R - C)
    d = d / (d.norm() + 1e-8)
    return d

def make_steer_hook(d, alpha: float):
    def hook(resid, hook):
        resid[:, -1, :] = resid[:, -1, :] + alpha * d
        return resid
    return hook

@torch.no_grad()
def sweep(model_name: str, prompts_path: str, layer: int, alphas):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)
    prompts = load_prompts(prompts_path)

    d = compute_direction(model, prompts, layer=layer)

    results = []
    for a in alphas:
        refusals = []
        for ex in prompts:
            toks = model.to_tokens(ex["prompt"])
            out = model.run_with_hooks(
                toks,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", make_steer_hook(d, a))],
            )
            # generate with hooks is more complex; simplest: just evaluate logits-based proxy here,
            # or re-run generate with a custom generate loop if needed.
        # (You’ll likely implement a small hooked-generate loop next.)
    return results
