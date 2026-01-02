import argparse
import json
import os
import csv
from typing import Dict, List, Tuple, Optional

import torch
from transformer_lens import HookedTransformer

from refusal_score import token_ids_first_piece, refusal_score_from_logits


# ----------------------------
# small robustness helper
# ----------------------------
def to_float(x):
    """Convert torch scalar or python number to float safely."""
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float(x)


# ----------------------------
# IO helpers
# ----------------------------
def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def maybe_load_scores_csv(path: str) -> Optional[Dict[str, float]]:
    """
    Returns dict: id -> s_inst if CSV exists and has those columns.
    """
    if not path or not os.path.exists(path):
        return None
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "id" not in row or "s_inst" not in row:
                continue
            try:
                out[row["id"]] = float(row["s_inst"])
            except Exception:
                pass
    return out if len(out) > 0 else None


# ----------------------------
# Prompt formatting
# ----------------------------
def chatml_wrap(tokenizer, user_prompt: str, system: str = "You are a helpful assistant.") -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ----------------------------
# Hooks
# ----------------------------
def add_direction_last_pos(direction: torch.Tensor, alpha: float, pos: int = -1):
    """
    Adds alpha * direction to act[:, pos, :].
    """
    def hook(act, hook):
        d = direction.to(device=act.device, dtype=act.dtype)
        act = act.clone()
        act[:, pos, :] = act[:, pos, :] + alpha * d
        return act
    return hook


@torch.no_grad()
def compute_refusal_score_for_prompt(
    model: HookedTransformer,
    tokenizer,
    prompt: str,
    ref_ids: torch.Tensor,
    ok_ids: torch.Tensor,
    system: str,
) -> float:
    full = chatml_wrap(tokenizer, prompt, system=system)
    toks = model.to_tokens(full)
    logits = model(toks)[0, -1]
    s = refusal_score_from_logits(logits, ref_ids, ok_ids)
    return to_float(s)


@torch.no_grad()
def get_activation_lastpos(
    model: HookedTransformer,
    tokenizer,
    prompt: str,
    hook_name: str,
    system: str,
) -> torch.Tensor:
    """
    Returns cache[hook_name][0, -1, :] as float32 tensor.
    """
    full = chatml_wrap(tokenizer, prompt, system=system)
    toks = model.to_tokens(full)
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook_name)
    if hook_name not in cache:
        raise RuntimeError(f"Hook name not found in cache: {hook_name}")
    act = cache[hook_name][0, -1, :].detach()
    return act.to(dtype=torch.float32)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out", default="runs/steering_balanced_qwen25.json")

    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--site", type=str, default="hook_resid_pre")
    ap.add_argument("--pos", type=int, default=-1)

    ap.add_argument("--top_n", type=int, default=30)
    ap.add_argument("--bottom_n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--scores_csv", default="runs/qwen25_scores.csv")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0])
    ap.add_argument("--max_n", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = HookedTransformer.from_pretrained("qwen2.5-0.5b", device=device, dtype=torch.bfloat16)
    inst = HookedTransformer.from_pretrained("qwen2.5-0.5b-instruct", device=device, dtype=torch.bfloat16)
    tok = base.tokenizer

    ref_ids = token_ids_first_piece(tok, ["I'm", "I", "Sorry", "I’m"])
    ok_ids  = token_ids_first_piece(tok, ["Sure", "Yes", "Here", "Of"])

    data = load_jsonl(args.prompts)
    if args.max_n and args.max_n > 0:
        data = data[:args.max_n]

    by_id = {ex["id"]: ex for ex in data if "id" in ex}
    edge = [ex for ex in data if ex.get("split") == "edge"]
    harmful = [ex for ex in data if ex.get("split") == "harmful"]

    if len(edge) == 0:
        raise RuntimeError("No edge prompts found in prompts.jsonl (split=='edge').")

    hook_name = f"blocks.{args.layer}.{args.site}"

    # ---- Step 1: pick balanced top/bottom edge prompts by s_inst ----
    scores_map = maybe_load_scores_csv(args.scores_csv)

    scored_edge: List[Tuple[str, float]] = []
    used_scores_source = None

    if scores_map is not None:
        for ex in edge:
            ex_id = ex["id"]
            if ex_id in scores_map:
                scored_edge.append((ex_id, float(scores_map[ex_id])))
        if len(scored_edge) >= (args.top_n + args.bottom_n):
            used_scores_source = args.scores_csv
        else:
            print(f"[warn] Only {len(scored_edge)} edge prompts had s_inst in {args.scores_csv}. "
                  f"Need >= {args.top_n + args.bottom_n}. Falling back to computing s_inst on-the-fly.")
            scored_edge = []

    if len(scored_edge) == 0:
        used_scores_source = "computed_on_the_fly"
        for ex in edge:
            ex_id = ex["id"]
            s_inst = compute_refusal_score_for_prompt(inst, tok, ex["prompt"], ref_ids, ok_ids, system=args.system)
            scored_edge.append((ex_id, s_inst))

    scored_edge.sort(key=lambda x: x[1])
    bottom = scored_edge[:args.bottom_n]
    top = scored_edge[-args.top_n:]

    bottom_ids = [ex_id for ex_id, _ in bottom]
    top_ids = [ex_id for ex_id, _ in top]

    if len(top_ids) < args.top_n or len(bottom_ids) < args.bottom_n:
        raise RuntimeError("Not enough edge prompts for balanced selection.")

    # ---- Step 2: build direction from activations in inst ----
    top_acts = []
    bot_acts = []

    for ex_id in top_ids:
        ex = by_id[ex_id]
        a = get_activation_lastpos(inst, tok, ex["prompt"], hook_name, system=args.system)
        top_acts.append(a)

    for ex_id in bottom_ids:
        ex = by_id[ex_id]
        a = get_activation_lastpos(inst, tok, ex["prompt"], hook_name, system=args.system)
        bot_acts.append(a)

    top_mean = torch.stack(top_acts, dim=0).mean(dim=0)
    bot_mean = torch.stack(bot_acts, dim=0).mean(dim=0)

    direction = (top_mean - bot_mean).to(device=device)
    direction_pre_norm = float(direction.norm().item())
    if direction_pre_norm < 1e-8:
        raise RuntimeError("Direction norm is ~0. Something went wrong.")
    direction = direction / direction.norm()
    direction_norm = float(direction.norm().item())

    # ---- Step 3: evaluate steering ----
    def eval_model(model: HookedTransformer, split_rows: List[dict], alpha: float) -> Tuple[float, float]:
        scores = []
        for ex in split_rows:
            full = chatml_wrap(tok, ex["prompt"], system=args.system)
            toks = model.to_tokens(full)

            logits = model.run_with_hooks(
                toks,
                fwd_hooks=[(hook_name, add_direction_last_pos(direction, alpha, pos=args.pos))],
            )[0, -1]

            s = refusal_score_from_logits(logits, ref_ids, ok_ids)
            scores.append(to_float(s))

        if len(scores) == 0:
            return 0.0, 0.0
        t = torch.tensor(scores, dtype=torch.float32)
        return float(t.mean().item()), float(t.std(unbiased=False).item())

    results = []
    for alpha in args.alphas:
        base_edge = eval_model(base, edge, alpha)
        base_harm = eval_model(base, harmful, alpha)
        inst_edge = eval_model(inst, edge, alpha)
        inst_harm = eval_model(inst, harmful, alpha)

        results.append({
            "alpha": float(alpha),
            "base_edge_meanstd": [base_edge[0], base_edge[1]],
            "base_harm_meanstd": [base_harm[0], base_harm[1]],
            "inst_edge_meanstd": [inst_edge[0], inst_edge[1]],
            "inst_harm_meanstd": [inst_harm[0], inst_harm[1]],
        })

        print(f"alpha={alpha:>6}: "
              f"base(edge)={base_edge[0]: .3f}±{base_edge[1]:.3f}  "
              f"base(harm)={base_harm[0]: .3f}±{base_harm[1]:.3f}  "
              f"inst(edge)={inst_edge[0]: .3f}±{inst_edge[1]:.3f}  "
              f"inst(harm)={inst_harm[0]: .3f}±{inst_harm[1]:.3f}")

    out = {
        "layer": args.layer,
        "site": args.site,
        "hook_name": hook_name,
        "pos": args.pos,
        "system": args.system,
        "alphas": [float(a) for a in args.alphas],
        "direction_norm": direction_norm,
        "direction_pre_norm": direction_pre_norm,
        "balanced_selection": {
            "method": "top/bottom by s_inst (edge prompts)",
            "top_n": args.top_n,
            "bottom_n": args.bottom_n,
            "scores_source": used_scores_source,
            "top_ids": top_ids,
            "bottom_ids": bottom_ids,
            "top_s_inst_minmax": [float(min(s for _, s in top)), float(max(s for _, s in top))],
            "bottom_s_inst_minmax": [float(min(s for _, s in bottom)), float(max(s for _, s in bottom))],
        },
        "counts": {"edge": len(edge), "harmful": len(harmful)},
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
