# Local LoRA (8GB GPU) — run order

Work from Git Bash or WSL (not plain PowerShell for `.sh` files).

## 1. One-time env setup (repo root)

```bash
cd /c/Sarastra/SilentTalk
bash setup_env.sh
```

This clones IndicTrans2, installs deps, and copies helper scripts into
`IndicTrans2/huggingface_interface/`.

## 2. Patch + download + data + train

```bash
cd /c/Sarastra/SilentTalk/IndicTrans2/huggingface_interface

python3 patch_files.py

# Option A (safer): export token, do not hardcode in file
export HF_TOKEN="hf_xxxxxxxx"
python3 download_model.py

# Option B: edit YOUR_HF_TOKEN inside download_model.py, run once, then remove the token line
python3 download_model.py

python3 prepare_data.py
bash run_train.sh
```

## OOM fallback (8GB)

If training OOMs, edit `run_train.sh`:

- `--batch_size 1`
- `--grad_accum_steps 32`
