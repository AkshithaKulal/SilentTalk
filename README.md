# SilentTalk

Local + Colab pipeline for ISL → Kannada/Tulu transfer learning (IndicTrans2 LoRA).

## Repo: code only

Large artifacts are **not** in git (see `.gitignore`):
- `model_cache/` (re-download from Hugging Face)
- `stage1_data/`, `stage2_data/`, `samanantar_en_kn.csv`
- `stage1_output/`, `stage2_output/`, `stage2_kn_tcy_output/` checkpoints

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

## Pipeline overview

| Stage | Direction | Data | Output |
|-------|-----------|------|--------|
| 1 | EN→Kannada (`eng_Latn`→`kan_Knda`) | Samanantar | `stage1_output/checkpoint-1500` |
| 2 (real) | KN→Tulu (`kan_Knda`→`brx_Deva` alias) | `kn_tcy_raw.csv` (organizers; no public mirror) | `stage2_kn_tcy_output/` |
| 2 (historical) | EN→Tulu-in-`kan_Knda` slot | synthetic ~19k | `stage2_output/checkpoint-3000` (BLEU ~0.84; incomplete) |

**Tulu tag:** IndicTrans2 has no `tcy_Knda`. Tulu uses stand-in tag `brx_Deva` with processor override to ISO `kn` (see `tulu_lang_alias.py`). Not real Bodo.

## Stage 2 (real KN→Tulu)

```powershell
# Drop DravidianLangTech KN–TCY as kn_tcy_raw.csv (columns: kannada,tulu)
python prepare_stage2_data.py --mode real
.\run_stage2_train.ps1
python test_stage2_inference.py
python test_stage2_quality.py
```

If real CSV is missing (degraded / non-competitive continuity only):

```powershell
python prepare_stage2_data.py --mode synthetic_degraded
.\run_stage2_train_synthetic.ps1
```

## Important scripts

| Script | Purpose |
|--------|---------|
| `train_lora.py` | LoRA training (Tulu alias override when `brx_Deva` is used) |
| `tulu_lang_alias.py` | `brx_Deva`↔Tulu mapping + `kn` processor override |
| `prepare_stage2_data.py` | `--mode real` or `synthetic_degraded` |
| `run_train.ps1` / `resume_train.ps1` | Stage 1 train / resume |
| `run_stage2_train.ps1` | Real KN→Tulu from Stage 1 ckpt-1500 |
| `run_stage2_train_synthetic.ps1` | Degraded EN→Tulu continuity |
| `test_inference.py` | Domain EN→KN check |
| `test_stage2_inference.py` / `test_stage2_quality.py` | KN→Tulu eval |
| `synthetic_en_tulu_fixed.csv` | Synthetic EN–Tulu (~19k) |
| `stage2_output/RESULTS.md` | Metrics + alias documentation |

## Security

Never commit HF tokens. Use `$env:HF_TOKEN` / `export HF_TOKEN=...`.

## ISL Demo: Predict To Spoken Tulu

For a full junior-friendly INCLUDE workflow (setup -> audit -> extraction -> training -> prediction -> speech), see:

`isl_recognition/INCLUDE_PIPELINE_RUNBOOK.md`

Prerequisite (from repo root):

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pip install -r isl_recognition\requirements.txt
```

Run sign prediction with mapped Kannada-script Tulu sentence:

```powershell
python isl_recognition\predict_tulu_speech.py --video isl_recognition\artifacts\thankyou_demo.mp4 --artifacts isl_recognition\artifacts --mapping isl_recognition\artifacts\tulu_sentence_map_kn.json --top-k 5
```

Enable speech output:

```powershell
python isl_recognition\predict_tulu_speech.py --video isl_recognition\artifacts\thankyou_demo.mp4 --artifacts isl_recognition\artifacts --mapping isl_recognition\artifacts\tulu_sentence_map_kn.json --top-k 5 --speak
```

Notes:
- If top-1 label is missing in mapping, the script falls back through top-k labels.
- If no mapped sentence is found, the script prints a warning and exits without crashing.
