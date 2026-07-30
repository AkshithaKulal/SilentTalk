# Fix / load synthetic_en_tulu.csv that has extra commas (Colab or local)
# Paste into Colab after the backtranslation finished.

import csv
from pathlib import Path

path = Path("/content/drive/MyDrive/SilentTalk/synthetic_en_tulu.csv")
fixed_path = Path("/content/drive/MyDrive/SilentTalk/synthetic_en_tulu_fixed.csv")

# --- 1) Robust load (handles extra commas by keeping only first split as EN, rest as Tulu) ---
rows = []
bad = 0
with open(path, encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for i, row in enumerate(reader, start=2):
        if len(row) == 2:
            eng, tulu = row[0].strip(), row[1].strip()
        elif len(row) > 2:
            # Extra commas in text: first field = english, join the rest as tulu
            # (or if eng had commas: everything except last is eng — we prefer eng first)
            eng, tulu = row[0].strip(), ",".join(row[1:]).strip()
            bad += 1
        else:
            bad += 1
            continue
        if eng and tulu:
            rows.append((eng, tulu))

print(f"Loaded {len(rows)} pairs | repaired/odd lines: {bad}")

# --- 2) Rewrite as proper quoted CSV (safe for pandas) ---
with open(fixed_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["english", "tulu"])
    writer.writerows(rows)

print(f"Wrote {fixed_path}")

# --- 3) Verify with pandas ---
import pandas as pd

df = pd.read_csv(fixed_path)
print(df.shape)
print(df.head(10))

import random
random.seed(42)
for i in random.sample(range(len(df)), min(10, len(df))):
    print(f"EN:   {df.iloc[i]['english']}")
    print(f"TULU: {df.iloc[i]['tulu']}")
    print()
