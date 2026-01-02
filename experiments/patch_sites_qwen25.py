import argparse, json, os
from collections import defaultdict
import torch
from transformer_lens import HookedTransformer
from refusal_score import token_ids_first_piece, refusal_score_from_logits

SITES = ["hook_resid_pre", "hook_attn_out", "hook_mlp_out", "hook_resid_post"]

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def chatml_wrap(tokenizer, user_prompt: str, system: str = "You are a helpful assistant.") -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def patch_last_pos(cached, pos=-1):
    def hook(act, hook):
        act[:, pos, :] = cached[:, pos, :]
        return act
    return hook

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out", default="runs/patch_sites_qwen25.json")
    ap.add_argument("--max_n", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = HookedTransformer.from_pretrained("qwen2.5-0.5b", device=device, dtype=torch.bfloat16)
    inst = HookedTransformer.from_pretrained("qwen2.5-0.5b-instruct", device=device, dtype=torch.bfloat16)
    tok = base.tokenizer

    ref_ids = token_ids_first_piece(tok, ["I'm", "I", "Sorry", "I’m"])
    ok_ids  = token_ids_first_piece(tok, ["Sure", "Yes", "Here", "Of"])

    data = load_jsonl(args.prompts)
    if args.max_n and args.max_n > 0:
        data = data[:args.max_n]

    n_layers = base.cfg.n_layers
    counts = defaultdict(int)

    # accumulate deltas
    acc = {site: {l: defaultdict(float) for l in range(n_layers)} for site in SITES}

    for ex in data:
        split = ex["split"]
        full = chatml_wrap(tok, ex["prompt"])
        toks = base.to_tokens(full)

        base_logits = base(toks)[0, -1]
        base_score = refusal_score_from_logits(base_logits, ref_ids, ok_ids)

        # cache all sites once from instruct
        _, cache = inst.run_with_cache(
            toks,
            names_filter=lambda n: any(n.endswith(site) for site in SITES),
        )

        for l in range(n_layers):
            for site in SITES:
                name = f"blocks.{l}.{site}"
                if name not in cache:
                    continue
                cached = cache[name]
                patched_logits = base.run_with_hooks(
                    toks,
                    fwd_hooks=[(name, patch_last_pos(cached, pos=-1))],
                )[0, -1]
                patched_score = refusal_score_from_logits(patched_logits, ref_ids, ok_ids)
                acc[site][l][split] += (patched_score - base_score)

        counts[split] += 1

    out = {"n_layers": n_layers, "counts": dict(counts), "delta_score_mean": {}}
    for site in SITES:
        out["delta_score_mean"][site] = {}
        for l in range(n_layers):
            out["delta_score_mean"][site][str(l)] = {}
            for split in counts:
                out["delta_score_mean"][site][str(l)][split] = acc[site][l][split] / max(counts[split], 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
