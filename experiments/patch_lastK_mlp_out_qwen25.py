import argparse
import json
import os
from collections import defaultdict

import torch
from transformer_lens import HookedTransformer

from refusal_score import token_ids_first_piece, refusal_score_from_logits


def to_float(x):
    """Convert torch scalar or python number to float safely."""
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float(x)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def chatml_wrap(tokenizer, user_prompt: str, system: str = "You are a helpful assistant.") -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def patch_last_pos(cached, pos=-1):
    """
    Overwrite act[:, pos, :] with cached[:, pos, :].
    For hook_mlp_out, this replaces that layer’s MLP write at that token position.
    """
    def hook(act, hook):
        act = act.clone()
        act[:, pos, :] = cached[:, pos, :]
        return act
    return hook

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out", default="runs/patch_lastK_mlp_out_qwen25.json")
    ap.add_argument("--max_n", type=int, default=0, help="0=no limit")
    ap.add_argument("--pos", type=int, default=-1)
    ap.add_argument("--system", default="You are a helpful assistant.")
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
    acc = {K: defaultdict(float) for K in range(1, n_layers + 1)}

    for ex in data:
        split = ex.get("split", "unknown")
        full = chatml_wrap(tok, ex["prompt"], system=args.system)
        toks = base.to_tokens(full)

        # baseline base score
        base_logits = base(toks)[0, -1]
        base_score = to_float(refusal_score_from_logits(base_logits, ref_ids, ok_ids))

        # cache ALL instruct hook_mlp_out
        _, cache = inst.run_with_cache(
            toks,
            names_filter=lambda n: n.endswith("hook_mlp_out"),
        )

        cached = {}
        for l in range(n_layers):
            name = f"blocks.{l}.hook_mlp_out"
            if name in cache:
                cached[l] = cache[name]

        # cumulative patching: patch last-K MLP outputs
        for K in range(1, n_layers + 1):
            layers_to_patch = list(range(n_layers - K, n_layers))
            hooks = []
            for l in layers_to_patch:
                if l not in cached:
                    continue
                hooks.append((f"blocks.{l}.hook_mlp_out", patch_last_pos(cached[l], pos=args.pos)))

            patched_logits = base.run_with_hooks(toks, fwd_hooks=hooks)[0, -1]
            patched_score = to_float(refusal_score_from_logits(patched_logits, ref_ids, ok_ids))

            acc[K][split] += (patched_score - base_score)

        counts[split] += 1

    out = {"n_layers": n_layers, "counts": dict(counts), "delta_score_mean": {}}
    for K in range(1, n_layers + 1):
        out["delta_score_mean"][str(K)] = {sp: acc[K][sp] / max(counts[sp], 1) for sp in counts}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", args.out)
    print("Example last-K means:")
    for K in [1, 2, 4, 8, 12, 16, 20, n_layers]:
        if str(K) in out["delta_score_mean"]:
            print(K, out["delta_score_mean"][str(K)])


if __name__ == "__main__":
    main()
