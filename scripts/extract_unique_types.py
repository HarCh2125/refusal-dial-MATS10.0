# Extract unique entries from the "type" field in the XSTest prompts CSV
import pandas as pd
IN_CSV = "data/xstest_prompts.csv"
OUT_TXT = "data/xstest_unique_types.txt"
# Read the CSV file
df = pd.read_csv(IN_CSV)
# Extract unique types
unique_types = df["type"].dropna().unique()
# Write unique types to a text file
with open(OUT_TXT, "w") as f:
    for t in sorted(unique_types):
        f.write(f"{t}\n")
print(f"Wrote {len(unique_types)} unique types to {OUT_TXT}")