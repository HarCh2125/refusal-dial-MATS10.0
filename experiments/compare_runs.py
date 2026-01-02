import json
from collections import defaultdict

def read(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def summarize(rows):
    s = defaultdict(lambda: {"n": 0, "ref": 0})
    for r in rows:
        sp = r.get("split", "unknown")
        s[sp]["n"] += 1
        s[sp]["ref"] += int(bool(r.get("refusal", False)))
    return s

def main(base_path, inst_path):
    base = summarize(read(base_path))
    inst = summarize(read(inst_path))

    splits = sorted(set(base.keys()) | set(inst.keys()))
    print("split\tbase_rate\tinst_rate\t(base_n, inst_n)")
    for sp in splits:
        bn, br = base[sp]["n"], base[sp]["ref"]
        in_, ir = inst[sp]["n"], inst[sp]["ref"]
        base_rate = br / bn if bn else 0.0
        inst_rate = ir / in_ if in_ else 0.0
        print(f"{sp}\t{base_rate:.3f}\t{inst_rate:.3f}\t({bn},{in_})")

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])