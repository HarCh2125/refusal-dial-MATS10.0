import argparse, json
from collections import defaultdict
import torch
from transformer_lens import HookedTransformer
from refusal_score import token_ids_first_piece, refusal_score_from_logits

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
    ap.add_argument("--out", default="runs/patch_lastK_qwen25.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = HookedTransformer.from_pretrained("qwen2.5-0.5b", device=device, dtype=torch.bfloat16)
    inst = HookedTransformer.from_pretrained("qwen2.5-0.5b-instruct", device=device, dtype=torch.bfloat16)
    tok = base.tokenizer

    ref_ids = token_ids_first_piece(tok, ["I'm", "I", "Sorry", "I’m"])
    ok_ids  = token_ids_first_piece(tok, ["Sure", "Yes", "Here", "Of"])

    data = load_jsonl(ap.parse_args().prompts)
    n_layers = base.cfg.n_layers
    counts = defaultdict(int)
    acc = {K: defaultdict(float) for K in range(1, n_layers+1)}

    for ex in data:
        split = ex["split"]
        full = chatml_wrap(tok, ex["prompt"])
        toks = base.to_tokens(full)

        base_score = refusal_score_from_logits(base(toks)[0, -1], ref_ids, ok_ids)

        _, cache = inst.run_with_cache(
            toks, names_filter=lambda n: n.endswith("hook_resid_pre")
        )

        # prebuild cached activations for all layers
        cached = {l: cache[f"blocks.{l}.hook_resid_pre"] for l in range(n_layers)}

        for K in range(1, n_layers+1):
            layers_to_patch = list(range(n_layers-K, n_layers))
            hooks = [(f"blocks.{l}.hook_resid_pre", patch_last_pos(cached[l], pos=-1)) for l in layers_to_patch]
            patched_logits = base.run_with_hooks(toks, fwd_hooks=hooks)[0, -1]
            patched_score = refusal_score_from_logits(patched_logits, ref_ids, ok_ids)
            acc[K][split] += (patched_score - base_score)

        counts[split] += 1

    out = {"n_layers": n_layers, "counts": dict(counts), "delta_score_mean": {}}
    for K in range(1, n_layers+1):
        out["delta_score_mean"][str(K)] = {sp: acc[K][sp]/max(counts[sp],1) for sp in counts}

    with open(ap.parse_args().out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", ap.parse_args().out)

if __name__ == "__main__":
    main()
