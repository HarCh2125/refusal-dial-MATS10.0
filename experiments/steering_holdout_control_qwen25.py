import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformer_lens import HookedTransformer

from refusal_score import token_ids_first_piece, refusal_score_from_logits


# ----------------------------
# IO helpers
# ----------------------------

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


def to_float(x) -> float:
    # refusal_score_from_logits might return float or tensor
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().item())
    return float(x)


def stable_hash01(s: str, seed: int = 0) -> float:
    # deterministic in [0,1)
    h = hashlib.md5((str(seed) + "::" + s).encode("utf-8")).hexdigest()
    v = int(h, 16) % 10_000_000
    return v / 10_000_000.0


def detect_col(header: List[str], candidates: List[str]) -> str:
    lower = {h.lower(): h for h in header}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"Could not find any of columns {candidates} in CSV header: {header}")


def load_scores_csv(path: str) -> Dict[str, dict]:
    """
    Expects a CSV with at least:
      - an id column (id / prompt_id / uid)
      - s_inst column (s_inst / inst / inst_score / score_inst)
    Optional:
      - split column (split)
    Returns: id -> dict(row)
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        if header is None:
            raise ValueError(f"Empty CSV: {path}")

        id_col = detect_col(header, ["id", "prompt_id", "uid"])
        s_inst_col = detect_col(header, ["s_inst", "inst", "inst_score", "score_inst"])

        split_col = None
        try:
            split_col = detect_col(header, ["split"])
        except KeyError:
            split_col = None

        for row in reader:
            pid = row[id_col]
            out[pid] = {
                "id": pid,
                "s_inst": float(row[s_inst_col]),
                "split": row[split_col] if split_col else None,
                "raw": row,
            }
    return out


def chatml_wrap(tokenizer, user_prompt: str, system: str = "You are a helpful assistant.") -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ----------------------------
# Core
# ----------------------------

@dataclass
class PromptEx:
    pid: str
    split: str
    prompt: str


def build_direction_from_ids(
    inst: HookedTransformer,
    tok,
    hook_name: str,
    pos: int,
    prompts_by_id: Dict[str, PromptEx],
    ids_top: List[str],
    ids_bottom: List[str],
    system: str,
    device: str,
) -> Tuple[torch.Tensor, float]:
    """
    direction = mean(act_top) - mean(act_bottom), using INST activations at hook_name and position pos.
    Returns:
      direction_unit: [d_model] float32 on device
      pre_norm: float
    """
    assert len(ids_top) > 0 and len(ids_bottom) > 0

    def mean_act(ids: List[str]) -> torch.Tensor:
        acc = None
        n = 0
        for pid in ids:
            ex = prompts_by_id[pid]
            full = chatml_wrap(tok, ex.prompt, system=system)
            toks = inst.to_tokens(full)

            _, cache = inst.run_with_cache(
                toks,
                names_filter=lambda n: n == hook_name,
            )
            act = cache[hook_name]  # [1, seq, d_model]
            vec = act[:, pos, :].squeeze(0).detach().to(torch.float32)  # [d_model]
            if acc is None:
                acc = vec.clone()
            else:
                acc += vec
            n += 1
        return acc / max(n, 1)

    top_mean = mean_act(ids_top)
    bot_mean = mean_act(ids_bottom)
    direction = top_mean - bot_mean  # [d_model]
    pre_norm = float(direction.norm().item())
    direction_unit = direction / (direction.norm() + 1e-8)
    return direction_unit.to(device), pre_norm


def add_vec_hook(direction_unit: torch.Tensor, alpha: float, pos: int):
    # direction_unit: [d_model] on same device; will cast to act dtype inside hook
    def hook(act, hook):
        # act: [batch, seq, d_model]
        act[:, pos, :] = act[:, pos, :] + (alpha * direction_unit).to(act.dtype)
        return act
    return hook


@torch.no_grad()
def eval_scores(
    model: HookedTransformer,
    tok,
    hook_name: str,
    pos: int,
    direction_unit: torch.Tensor,
    alphas: List[float],
    eval_ids: List[str],
    prompts_by_id: Dict[str, PromptEx],
    ref_ids: List[int],
    ok_ids: List[int],
    system: str,
) -> Dict[float, Dict[str, object]]:
    """
    Returns per-alpha:
      - scores: list[float] aligned with eval_ids
      - refuse_flags: list[int] (1 if score>0)
      - mean, std, refusal_rate
    """
    out = {}
    for a in alphas:
        scores = []
        flags = []
        for pid in eval_ids:
            ex = prompts_by_id[pid]
            full = chatml_wrap(tok, ex.prompt, system=system)
            toks = model.to_tokens(full)

            if abs(a) < 1e-12:
                logits = model(toks)[0, -1]
            else:
                logits = model.run_with_hooks(
                    toks,
                    fwd_hooks=[(hook_name, add_vec_hook(direction_unit, a, pos=pos))],
                )[0, -1]

            s = to_float(refusal_score_from_logits(logits, ref_ids, ok_ids))
            scores.append(s)
            flags.append(1 if s > 0 else 0)

        t = torch.tensor(scores, dtype=torch.float32)
        out[a] = {
            "mean": float(t.mean().item()) if len(scores) else 0.0,
            "std": float(t.std(unbiased=False).item()) if len(scores) else 0.0,
            "refusal_rate": float(sum(flags) / max(len(flags), 1)),
            "scores": scores,
            "refuse_flags": flags,
        }
    return out


def write_per_example_csv(
    out_csv: str,
    run_name: str,
    results: dict,
    eval_sets: Dict[str, List[str]],
):
    """
    results: dict like:
      results["base"]["edge_train"][alpha]["scores"] aligned with eval_sets["edge_train"]
    """
    ensure_dir(out_csv)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "model", "eval_set", "alpha", "id", "score", "refuse_flag"])
        for model_name in ["base", "inst"]:
            for set_name, ids in eval_sets.items():
                for alpha_str, payload in results[model_name][set_name].items():
                    a = float(alpha_str)
                    scores = payload["scores"]
                    flags = payload["refuse_flags"]
                    assert len(scores) == len(ids) == len(flags)
                    for pid, s, fl in zip(ids, scores, flags):
                        w.writerow([run_name, model_name, set_name, a, pid, s, fl])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--scores_csv", default="runs/qwen25_scores.csv")

    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--site", type=str, required=True)  # e.g. hook_resid_pre
    ap.add_argument("--pos", type=int, default=-1)

    ap.add_argument("--top_n", type=int, default=30)
    ap.add_argument("--bottom_n", type=int, default=30)

    ap.add_argument("--alphas", type=float, nargs="+", required=True)

    ap.add_argument("--holdout_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--system", type=str, default="You are a helpful assistant.")

    ap.add_argument("--out_json", default="runs/steering_holdout_control_qwen25.json")
    ap.add_argument("--out_csv", default="runs/steering_holdout_control_qwen25_scores.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = HookedTransformer.from_pretrained("qwen2.5-0.5b", device=device, dtype=torch.bfloat16)
    inst = HookedTransformer.from_pretrained("qwen2.5-0.5b-instruct", device=device, dtype=torch.bfloat16)
    tok = base.tokenizer

    hook_name = f"blocks.{args.layer}.{args.site}"

    # tokens for refusal_score
    ref_ids = token_ids_first_piece(tok, ["I'm", "I", "Sorry", "I’m"])
    ok_ids = token_ids_first_piece(tok, ["Sure", "Yes", "Here", "Of"])

    # load prompts
    rows = load_jsonl(args.prompts)
    prompts_by_id: Dict[str, PromptEx] = {}
    for r in rows:
        pid = r.get("id", None) or r.get("prompt_id", None) or r.get("uid", None)
        if pid is None:
            raise KeyError("prompts.jsonl needs an 'id' (or prompt_id/uid) field per line.")
        split = r.get("split", "unknown")
        prompt = r.get("prompt", None) or r.get("text", None)
        if prompt is None:
            raise KeyError("prompts.jsonl needs a 'prompt' (or text) field per line.")
        prompts_by_id[pid] = PromptEx(pid=pid, split=split, prompt=prompt)

    # load scores for ranking (s_inst)
    scores = load_scores_csv(args.scores_csv)

    # collect edge IDs
    edge_ids = [pid for pid, ex in prompts_by_id.items() if ex.split == "edge"]
    harmful_ids = [pid for pid, ex in prompts_by_id.items() if ex.split == "harmful"]

    edge_ids.sort()

    # deterministic holdout split
    edge_train, edge_holdout = [], []
    for pid in edge_ids:
        u = stable_hash01(pid, seed=args.seed)
        if u < args.holdout_frac:
            edge_holdout.append(pid)
        else:
            edge_train.append(pid)

    # rank only within edge_train using s_inst
    ranked = []
    for pid in edge_train:
        if pid not in scores:
            raise KeyError(f"ID {pid} not found in scores CSV {args.scores_csv}")
        ranked.append((scores[pid]["s_inst"], pid))
    ranked.sort(key=lambda x: x[0], reverse=True)

    top_ids = [pid for _, pid in ranked[:args.top_n]]
    bottom_ids = [pid for _, pid in ranked[-args.bottom_n:]]

    # build direction from instruct activations on TOP/BOTTOM (edge_train only)
    direction_unit, pre_norm = build_direction_from_ids(
        inst=inst,
        tok=tok,
        hook_name=hook_name,
        pos=args.pos,
        prompts_by_id=prompts_by_id,
        ids_top=top_ids,
        ids_bottom=bottom_ids,
        system=args.system,
        device=device,
    )
    direction_norm = float(direction_unit.norm().item())

    # evaluation sets
    eval_sets = {
        "edge_train": edge_train,
        "edge_holdout": edge_holdout,
        "harmful": harmful_ids,
    }

    # eval: base and inst, on each eval set, for each alpha
    results = {"base": {}, "inst": {}}
    for model_name, model in [("base", base), ("inst", inst)]:
        results[model_name] = {}
        for set_name, ids in eval_sets.items():
            per_alpha = eval_scores(
                model=model,
                tok=tok,
                hook_name=hook_name,
                pos=args.pos,
                direction_unit=direction_unit,
                alphas=args.alphas,
                eval_ids=ids,
                prompts_by_id=prompts_by_id,
                ref_ids=ref_ids,
                ok_ids=ok_ids,
                system=args.system,
            )
            # json keys must be strings
            results[model_name][set_name] = {str(k): v for k, v in per_alpha.items()}

    run_name = f"holdout_layer{args.layer}_{args.site}"

    # write per-example CSV
    write_per_example_csv(args.out_csv, run_name, results, eval_sets)

    # write summary JSON (keep per-example data out of JSON to avoid massive files)
    summary = {
        "run": run_name,
        "layer": args.layer,
        "site": args.site,
        "hook_name": hook_name,
        "pos": args.pos,
        "system": args.system,
        "alphas": [float(a) for a in args.alphas],
        "holdout": {
            "seed": args.seed,
            "holdout_frac": args.holdout_frac,
            "counts": {
                "edge_total": len(edge_ids),
                "edge_train": len(edge_train),
                "edge_holdout": len(edge_holdout),
                "harmful": len(harmful_ids),
            },
        },
        "direction": {
            "method": "top/bottom by s_inst within EDGE-TRAIN only",
            "top_n": args.top_n,
            "bottom_n": args.bottom_n,
            "top_ids": top_ids,
            "bottom_ids": bottom_ids,
            "direction_norm": direction_norm,
            "direction_pre_norm": pre_norm,
            "scores_source": args.scores_csv,
        },
        "results_summary": {
            # keep only mean/std/refusal_rate
            model: {
                set_name: {
                    a: {
                        "mean": payload["mean"],
                        "std": payload["std"],
                        "refusal_rate": payload["refusal_rate"],
                    }
                    for a, payload in per_alpha.items()
                }
                for set_name, per_alpha in model_block.items()
            }
            for model, model_block in results.items()
        },
        "outputs": {
            "out_json": args.out_json,
            "out_csv": args.out_csv,
        },
    }

    ensure_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", args.out_json)
    print("Wrote:", args.out_csv)
    print("Counts:", summary["holdout"]["counts"])
    print("Direction pre-norm:", pre_norm)


if __name__ == "__main__":
    main()
