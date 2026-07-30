import os
import pandas as pd
from datasets import load_dataset

samanantar_kn = load_dataset("ai4bharat/samanantar", "kn")
samanantar_kn["train"].to_csv("samanantar_en_kn.csv")

df = pd.read_csv("samanantar_en_kn.csv").dropna(subset=["src", "tgt"])
df_sample = df.sample(n=300000, random_state=42).reset_index(drop=True)

os.makedirs("stage1_data/train/eng_Latn-kan_Knda", exist_ok=True)
os.makedirs("stage1_data/dev/eng_Latn-kan_Knda", exist_ok=True)

train_df = df_sample.sample(frac=0.98, random_state=42)
dev_df = df_sample.drop(train_df.index).sample(n=500, random_state=42)  # small dev set from the start

train_df["src"].to_csv("stage1_data/train/eng_Latn-kan_Knda/train.eng_Latn", index=False, header=False)
train_df["tgt"].to_csv("stage1_data/train/eng_Latn-kan_Knda/train.kan_Knda", index=False, header=False)
dev_df["src"].to_csv("stage1_data/dev/eng_Latn-kan_Knda/dev.eng_Latn", index=False, header=False)
dev_df["tgt"].to_csv("stage1_data/dev/eng_Latn-kan_Knda/dev.kan_Knda", index=False, header=False)

print(f"Train: {len(train_df)} | Dev: {len(dev_df)}")
