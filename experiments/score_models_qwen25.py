import argparse, json, os
from collections import defaultdict
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from refusal_score import token_ids_first_piece, refusal_score_from_logits

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def chatml_wrap(tokenizer, user_prompt: str, system: str = "You are a helpful assistant.") -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out_csv", default="runs/qwen25_scores.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = HookedTransformer.from_pretrained("qwen2.5-0.5b", device=device, dtype=torch.bfloat16)
    inst = HookedTransformer.from_pretrained("qwen2.5-0.5b-instruct", device=device, dtype=torch.bfloat16)
    tok = base.tokenizer

    ref_ids = token_ids_first_piece(tok, ["I'm", "I", "Sorry", "I’m"])
    ok_ids  = token_ids_first_piece(tok, ["Sure", "Yes", "Here", "Of"])

    rows = []
    for ex in load_jsonl(args.prompts):
        full = chatml_wrap(tok, ex["prompt"])
        toks = base.to_tokens(full)

        base_logits = base(toks)[0, -1]
        inst_logits = inst(toks)[0, -1]

        s_base = refusal_score_from_logits(base_logits, ref_ids, ok_ids)
        s_inst = refusal_score_from_logits(inst_logits, ref_ids, ok_ids)

        rows.append({
            "id": ex["id"],
            "split": ex["split"],
            "category": ex.get("category",""),
            "s_base": s_base,
            "s_inst": s_inst,
            "s_gap": (s_inst - s_base),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("Saved:", args.out_csv)
    print(df.groupby("split")[["s_base","s_inst","s_gap"]].mean())

if __name__ == "__main__":
    main()
