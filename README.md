# SilentTalk

Local + Colab pipeline for ISL → Kannada/Tulu transfer learning (IndicTrans2 LoRA).

## Repo: code only

Large artifacts are **not** in git (see `.gitignore`):
- `model_cache/` (re-download from Hugging Face)
- `stage1_data/`, `stage2_data/`, `samanantar_en_kn.csv`
- `stage1_output/`, `stage2_output/` checkpoints

Copy those separately (Drive/USB) to the 56GB machine if needed, or rebuild.

## Clone on the other machine

```bash
git clone https://github.com/AkshithaKulal/SilentTalk.git
cd SilentTalk/IndicTrans2/huggingface_interface
```

## Setup (Windows PowerShell)

```powershell
$env:HF_TOKEN = "hf_xxxxxxxx"   # your Hugging Face token
python -m pip install -e IndicTransToolkit_repo
python -m pip install "transformers==4.53.2" "peft==0.15.2" "torchao==0.9.0" sacrebleu datasets accelerate sentencepiece
# Install torch with CUDA matching that machine
```

Download base model:

```powershell
python download_model.py
```

## Important scripts

| Script | Purpose |
|--------|---------|
| `train_lora.py` | LoRA training (patched) |
| `run_train.ps1` / `resume_train.ps1` | Stage 1 train / resume |
| `test_inference.py` | Domain EN→KN check |
| `synthetic_en_tulu_fixed.csv` | Synthetic EN–Tulu (~19k) |
| `resume_stage2_local.py` / `run_stage2_local.py` | Stage 2 local resume helpers |

## Security

Never commit HF tokens. Use `$env:HF_TOKEN` / `export HF_TOKEN=...`.
