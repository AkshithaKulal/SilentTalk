import os
import shutil
import json

BASE_DIR = r"C:\Sarastra\SilentTalk\IndicTrans2\huggingface_interface"
os.chdir(BASE_DIR)

# ---- Step 1: Locate your latest Stage 2 checkpoint ----
# Adjust this path to wherever your Stage 2 checkpoint actually is locally
# (download it from Drive first if it's only in Colab right now)
checkpoint_path = os.path.join(BASE_DIR, "stage2_output", "checkpoint-1500")
base_model_path = os.path.join(BASE_DIR, "model_cache", "indictrans2-en-indic-1B")

if not os.path.exists(checkpoint_path):
    print(f"WARNING: {checkpoint_path} not found locally.")
    print("Download this folder from Google Drive (SilentTalk/stage2_output/checkpoint-1500)")
    print("into the same path locally before running training.")
else:
    print(f"Found checkpoint at {checkpoint_path}")

# ---- Step 2: Fix the stale tokenizer files (same fix as Colab) ----
files_to_copy = [
    "tokenization_indictrans.py",
    "configuration_indictrans.py",
    "modeling_indictrans.py",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
for fname in files_to_copy:
    src = os.path.join(base_model_path, fname)
    dst = os.path.join(checkpoint_path, fname)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {fname}")

# ---- Step 3: Fix adapter_config.json to point at the correct local base model path ----
adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
if os.path.exists(adapter_config_path):
    with open(adapter_config_path) as f:
        config = json.load(f)
    print("Current base_model_name_or_path:", config.get("base_model_name_or_path"))
    config["base_model_name_or_path"] = base_model_path
    with open(adapter_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("Updated base_model_name_or_path to:", base_model_path)
else:
    print(f"WARNING: {adapter_config_path} not found — skip adapter_config fix until checkpoint exists.")

# ---- Step 4: Clear any stale cached tokenizer module ----
cache_path = os.path.expanduser(r"~\.cache\huggingface\modules\transformers_modules\checkpoint-1500")
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
    print("Cleared stale cached module")

print("\nSetup complete. Ready to launch training.")
