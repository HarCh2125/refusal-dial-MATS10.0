import argparse, json, os
from collections import defaultdict

import torch
from transformer_lens import HookedTransformer

from refusal_score import token_ids_first_piece, refusal_score_from_logits


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def chatml_wrap(tokenizer, user_prompt: str, system: str = "You are a helpful assistant.") -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def patch_last_pos_from_cache(cached_act, pos=-1):
    # returns a hook fn that overwrites act[:, pos, :] with cached_act[:, pos, :]
    def hook(act, hook):
        act[:, pos, :] = cached_act[:, pos, :]
        return act
    return hook

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out", default="runs/patch_resid_pre_qwen25.json")
    ap.add_argument("--max_n", type=int, default=0, help="0=no limit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = HookedTransformer.from_pretrained("qwen2.5-0.5b", device=device, dtype=torch.bfloat16)
    inst = HookedTransformer.from_pretrained("qwen2.5-0.5b-instruct", device=device, dtype=torch.bfloat16)

    tok = base.tokenizer  # should match
    ref_ids = token_ids_first_piece(tok, ["I'm", "I", "Sorry", "I’m"])
    ok_ids  = token_ids_first_piece(tok, ["Sure", "Yes", "Here", "Of"])

    data = load_jsonl(args.prompts)
    if args.max_n and args.max_n > 0:
        data = data[:args.max_n]

    n_layers = base.cfg.n_layers

    # store per-layer average delta score (patched - base)
    per_layer = {l: defaultdict(float) for l in range(n_layers)}
    counts = defaultdict(int)

    for ex in data:
        split = ex.get("split", "unknown")

        # Use identical ChatML formatting for both models
        full_prompt = chatml_wrap(tok, ex["prompt"])
        toks = base.to_tokens(full_prompt)

        # Baseline base score (single forward pass)
        base_logits = base(toks)[0, -1]  # logits at next-token position
        base_score = refusal_score_from_logits(base_logits, ref_ids, ok_ids)

        # Run instruct once + cache all resid_pre
        _, cache = inst.run_with_cache(
            toks,
            names_filter=lambda n: n.endswith("hook_resid_pre"),
        )

        for l in range(n_layers):
            name = f"blocks.{l}.hook_resid_pre"
            cached = cache[name]  # [1, seq, d_model]

            patched_logits = base.run_with_hooks(
                toks,
                fwd_hooks=[(name, patch_last_pos_from_cache(cached, pos=-1))],
            )[0, -1]

            patched_score = refusal_score_from_logits(patched_logits, ref_ids, ok_ids)
            delta = patched_score - base_score

            per_layer[l][split] += delta

        counts[split] += 1

    # average
    out = {"n_layers": n_layers, "counts": dict(counts), "delta_score_mean": {}}
    for l in range(n_layers):
        out["delta_score_mean"][str(l)] = {}
        for split, tot in per_layer[l].items():
            out["delta_score_mean"][str(l)][split] = tot / max(counts[split], 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote patching summary to {args.out}")
    print("Top layers by harmful delta (mean):")
    harmful = [(int(l), out["delta_score_mean"][l].get("harmful", 0.0)) for l in out["delta_score_mean"]]
    harmful.sort(key=lambda x: x[1], reverse=True)
    print(harmful[:8])

if __name__ == "__main__":
    main()
