import argparse, json, os
import numpy as np
import torch
from transformer_lens import HookedTransformer

from refusal_score import token_ids_first_piece, refusal_score_from_logits

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def chatml_wrap(tokenizer, user_prompt: str, system: str = "You are a helpful assistant.") -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def add_dir_hook(direction, alpha, pos=-1):
    # direction: [d_model]
    def hook(act, hook):
        act[:, pos, :] = act[:, pos, :] + alpha * direction
        return act
    return hook

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--site", default="hook_resid_pre", choices=["hook_resid_pre", "hook_resid_post"])
    ap.add_argument("--alphas", default="-6,-4,-2,-1,0,1,2,4,6")
    ap.add_argument("--out", default="runs/steering_qwen25.json")
    ap.add_argument("--refusal_threshold", type=float, default=0.0,
                    help="Label edge prompts as 'refuse' if s_inst > threshold")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = HookedTransformer.from_pretrained("qwen2.5-0.5b", device=device, dtype=torch.bfloat16)
    inst = HookedTransformer.from_pretrained("qwen2.5-0.5b-instruct", device=device, dtype=torch.bfloat16)
    tok = base.tokenizer

    ref_ids = token_ids_first_piece(tok, ["I'm", "I", "Sorry", "I’m"])
    ok_ids  = token_ids_first_piece(tok, ["Sure", "Yes", "Here", "Of"])

    data = load_jsonl(args.prompts)
    edge = [ex for ex in data if ex["split"] == "edge"]
    harmful = [ex for ex in data if ex["split"] == "harmful"]

    # ----- 1) Build refusal direction using edge prompts only -----
    # Label edge prompts by instruct refusal_score (logits-based; no generation)
    refuse_vecs = []
    comply_vecs = []

    hook_name = f"blocks.{args.layer}.{args.site}"

    for ex in edge:
        full = chatml_wrap(tok, ex["prompt"])
        toks = inst.to_tokens(full)

        # get instruct logits score + cache chosen site
        logits, cache = inst.run_with_cache(toks, names_filter=lambda n: n == hook_name)
        s_inst = refusal_score_from_logits(logits[0, -1], ref_ids, ok_ids)

        act = cache[hook_name][0, -1]  # [d_model]

        if s_inst > args.refusal_threshold:
            refuse_vecs.append(act)
        else:
            comply_vecs.append(act)

    refuse_mean = torch.stack(refuse_vecs).mean(dim=0)
    comply_mean = torch.stack(comply_vecs).mean(dim=0)

    direction = (refuse_mean - comply_mean)
    direction = direction / (direction.norm() + 1e-8)

    print(f"Built direction from edge prompts at {hook_name}")
    print(f"edge labels: refuse={len(refuse_vecs)} comply={len(comply_vecs)}")

    # ----- 2) Evaluate steering on both models (scores only; safe) -----
    alphas = [float(x) for x in args.alphas.split(",")]

    def eval_model(model, subset, alpha):
        scores = []
        for ex in subset:
            full = chatml_wrap(tok, ex["prompt"])
            toks = model.to_tokens(full)
            logits = model.run_with_hooks(
                toks, fwd_hooks=[(hook_name, add_dir_hook(direction, alpha, pos=-1))]
            )
            s = refusal_score_from_logits(logits[0, -1], ref_ids, ok_ids)
            scores.append(s)
        return float(np.mean(scores)), float(np.std(scores))

    out = {
        "layer": args.layer,
        "site": args.site,
        "alphas": alphas,
        "direction_norm": float(direction.norm().item()),
        "edge_counts": {"refuse": len(refuse_vecs), "comply": len(comply_vecs)},
        "results": []
    }

    for a in alphas:
        row = {
            "alpha": a,
            "base_edge_meanstd": eval_model(base, edge, a),
            "base_harm_meanstd": eval_model(base, harmful, a),
            "inst_edge_meanstd": eval_model(inst, edge, a),
            "inst_harm_meanstd": eval_model(inst, harmful, a),
        }
        out["results"].append(row)
        print("alpha", a, "base_harm_mean", row["base_harm_meanstd"][0], "inst_harm_mean", row["inst_harm_meanstd"][0])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
