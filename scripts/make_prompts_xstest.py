import argparse
import json
import os
from typing import List, Dict

import pandas as pd


def _norm(s: str) -> str:
    return str(s).strip().lower()


def load_xstest(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, keep_default_na=False)

    required = {"id", "prompt", "type", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}. Found: {list(df.columns)}")

    # Normalize label
    df["label"] = df["label"].map(_norm)
    df["type"] = df["type"].astype(str).str.strip()

    # Basic validation
    bad = df[~df["label"].isin(["safe", "unsafe"])]
    if len(bad) > 0:
        raise ValueError(f"Found labels other than safe/unsafe. Examples:\n{bad[['id','label']].head()}")

    return df


def df_to_records(
    df: pd.DataFrame,
    make_benign: int = 0,
    seed: int = 0,
    max_harmful: int = 0,
    max_edge: int = 0,
) -> List[Dict]:
    # harmful = unsafe
    harmful_df = df[df["label"] == "unsafe"].copy()

    # edge = safe
    safe_df = df[df["label"] == "safe"].copy()

    # Carving out a "benign" subset from safe prompts
    benign_df = pd.DataFrame(columns=df.columns)
    edge_df = safe_df

    if make_benign and make_benign > 0:
        benign_df = safe_df.sample(n=min(make_benign, len(safe_df)), random_state=seed)
        edge_df = safe_df.drop(index=benign_df.index)

    if max_harmful and max_harmful > 0:
        harmful_df = harmful_df.sample(n=min(max_harmful, len(harmful_df)), random_state=seed)

    if max_edge and max_edge > 0:
        edge_df = edge_df.sample(n=min(max_edge, len(edge_df)), random_state=seed)

    records = []

    # Helper to keep ID format stable
    def add_rows(prefix: str, split: str, df_part: pd.DataFrame):
        for k, row in enumerate(df_part.itertuples(index=False), start=1):
            records.append({
                "id": f"{prefix}{k:03d}",
                "split": split,
                "category": getattr(row, "type"),
                "prompt": getattr(row, "prompt"),

                # Keep provenance (helps debugging and later analysis)
                "xstest_id": int(getattr(row, "id")),
                "xstest_type": getattr(row, "type"),
                "xstest_label": getattr(row, "label"),

                # These columns exist in your CSV; keep if present
                "focus": getattr(row, "focus", ""),
                "note": getattr(row, "note", ""),
            })

    add_rows("h", "harmful", harmful_df)
    add_rows("e", "edge", edge_df)
    if len(benign_df) > 0:
        add_rows("b", "benign", benign_df)

    return records


def write_jsonl(records: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/xstest_prompts.csv")
    ap.add_argument("--out_jsonl", default="data/prompts.jsonl")
    ap.add_argument("--make_benign", type=int, default=0,
                    help="Optional: sample this many SAFE prompts into split=benign; remaining SAFE go to edge.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_harmful", type=int, default=0, help="Optional cap for quick runs (0 = no cap)")
    ap.add_argument("--max_edge", type=int, default=0, help="Optional cap for quick runs (0 = no cap)")
    args = ap.parse_args()

    df = load_xstest(args.in_csv)
    records = df_to_records(
        df,
        make_benign=args.make_benign,
        seed=args.seed,
        max_harmful=args.max_harmful,
        max_edge=args.max_edge,
    )
    write_jsonl(records, args.out_jsonl)

    # Summary only (no prompt text)
    splits = {}
    for r in records:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print(f"Wrote {len(records)} records to {args.out_jsonl}. Split counts: {splits}")


if __name__ == "__main__":
    main()
