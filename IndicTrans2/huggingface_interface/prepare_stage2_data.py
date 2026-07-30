import os

# This assumes you'll drop the DravidianLangTech data in as kn_tcy_raw.csv
# with columns: kannada, tulu  — once you receive it, run this script

import pandas as pd

if not os.path.exists("kn_tcy_raw.csv"):
    print("kn_tcy_raw.csv not found yet — waiting on DravidianLangTech dataset from organizers.")
    print("Once received, save it here as kn_tcy_raw.csv with columns: kannada, tulu")
else:
    df = pd.read_csv("kn_tcy_raw.csv").dropna()
    print(f"Loaded {len(df)} Kannada-Tulu pairs")

    os.makedirs("stage2_data/train/kan_Knda-tcy_Knda", exist_ok=True)
    os.makedirs("stage2_data/dev/kan_Knda-tcy_Knda", exist_ok=True)

    train_df = df.sample(frac=0.9, random_state=42)
    dev_df = df.drop(train_df.index)

    train_df["kannada"].to_csv(
        "stage2_data/train/kan_Knda-tcy_Knda/train.kan_Knda", index=False, header=False
    )
    train_df["tulu"].to_csv(
        "stage2_data/train/kan_Knda-tcy_Knda/train.tcy_Knda", index=False, header=False
    )
    dev_df["kannada"].to_csv(
        "stage2_data/dev/kan_Knda-tcy_Knda/dev.kan_Knda", index=False, header=False
    )
    dev_df["tulu"].to_csv(
        "stage2_data/dev/kan_Knda-tcy_Knda/dev.tcy_Knda", index=False, header=False
    )

    print(f"Train: {len(train_df)} | Dev: {len(dev_df)}")
