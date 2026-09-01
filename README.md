# SilentTalk

ISL sign recognition → English gloss → Kannada translation → speech (Sarvam / Parler / MMS).

---

## Live app — quick start (clone → setup → run)

### Prerequisites

| Tool | Version | Notes |
|------|---------|--------|
| **Python** | 3.10+ | [python.org](https://python.org) |
| **Node.js** | 18+ | Auto-installed on Windows via `winget` during setup |
| **MSVC Build Tools** | — | Auto-installed on Windows (needed for IndicTransToolkit) |
| **Git** | any | Clone the repo |
| **NVIDIA GPU** | 6–8 GB VRAM | Recommended (translate + optional Parler TTS) |
| **HF token** | — | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| **Sarvam API key** | optional | Best Kannada TTS — [dashboard.sarvam.ai](https://dashboard.sarvam.ai) |

### 1. Clone

```powershell
git clone https://github.com/AkshithaKulal/SilentTalk.git
cd SilentTalk
```

### 2. Setup (one time)

**Windows (recommended — run PowerShell as Administrator on a fresh PC):**

```powershell
.\setup.ps1
```

Skip automatic Node.js / MSVC install (manual prereqs):

```powershell
.\setup.ps1 -SkipSystemDeps
```

**Or cross-platform:**

```powershell
python setup.py
```

This will:

- Auto-install **Node.js LTS** and **Microsoft C++ Build Tools** on Windows (via `winget`)
- Create `silent-venv/` and install Python packages (`requirements-app.txt`)
- Install PyTorch with CUDA 12.1 (if needed)
- Install IndicTransToolkit
- Download MediaPipe hand/pose models
- Download LoRA adapter if missing (`checkpoint-1500-inference/`)
- Run `npm install` + `npm run build` for the React UI

### 3. Configure `.env`

Copy template and edit (never commit `.env`):

```env
HF_TOKEN=hf_xxxxxxxx
SARVAM_API_KEY=sk_xxxxxxxx
```

- **HF_TOKEN** — required on first run (downloads IndicTrans2 + TTS models)
- **SARVAM_API_KEY** — optional; best Kannada pronunciation (Bulbul v3)

### 4. Run

**Production (laptop / demo — one port):**

```powershell
.\silent-venv\Scripts\Activate.ps1
python app.py
```

Open **http://localhost:5000**

**Phone camera on same Wi-Fi (HTTPS required):**

```powershell
python app.py --https
```

Open **https://YOUR_LAN_IP:5000** on phone (accept certificate warning).

**Development (UI hot-reload):**

```powershell
# Terminal 1
.\scripts\dev.ps1 backend

# Terminal 2
.\scripts\dev.ps1 frontend
```

Open **http://localhost:5173** (not 5000 — Vite proxies `/api` to backend).

### 5. Rebuild UI after frontend changes

```powershell
cd frontend
npm run build
cd ..
python app.py
```

---

## What gets installed

| Component | Package / source |
|-----------|------------------|
| Web API | FastAPI + Uvicorn |
| Sign model | `isl_recognition/transfer_pack/sign_bilstm.pt` (in git) |
| Landmarks | MediaPipe (CPU) |
| Translation | IndicTrans2 + LoRA `checkpoint-1500-inference/` |
| TTS Tier 1 | Sarvam Bulbul (cloud, `.env`) |
| TTS Tier 2 | Indic Parler (local GPU, optional) |
| TTS fallback | MMS Kannada (`facebook/mms-tts-kan`) |

**First run downloads** (~4–8 GB total from Hugging Face): translation base model, MMS TTS, optionally Parler.

---

## ISL training only (office PC)

For landmark extraction + BiLSTM training (not needed to run the live app):

```powershell
.\silent-venv\Scripts\Activate.ps1
python -m pip install -r isl_recognition\requirements.txt
```

Full INCLUDE workflow: `isl_recognition/INCLUDE_PIPELINE_RUNBOOK.md`

---

## Translation / LoRA training (IndicTrans2)

Large artifacts are **not** in git (see `.gitignore`):

- `model_cache/`, `stage1_data/`, `stage2_data/`
- Training checkpoints under `stage1_output/`, `stage2_output/`

### Clone for training setup

```powershell
git clone https://github.com/AkshithaKulal/SilentTalk.git
cd SilentTalk
.\setup_env.ps1
```

Then follow `IndicTrans2/huggingface_interface/README.md`.

### Pipeline overview

| Stage | Direction | Data | Output |
|-------|-----------|------|--------|
| 1 | EN→Kannada | Samanantar | `stage1_output/checkpoint-1500` |
| 2 (real) | KN→Tulu | `kn_tcy_raw.csv` | `stage2_kn_tcy_output/` |
| 2 (synthetic) | EN→Tulu | synthetic ~19k | `stage2_output/checkpoint-3000` |

**Tulu tag:** stand-in `brx_Deva` with processor override — see `tulu_lang_alias.py`.

### Important training scripts

| Script | Purpose |
|--------|---------|
| `train_lora.py` | LoRA training |
| `prepare_stage2_data.py` | `--mode real` or `synthetic_degraded` |
| `run_train.ps1` / `run_stage2_train.ps1` | Stage 1 / 2 training |

## Security

Never commit `.env`, API keys, or HF tokens.

```powershell
$env:HF_TOKEN = "hf_xxx"   # or use .env file
```

---

## Legacy CLI demo (Tulu speech script)

```powershell
python isl_recognition\predict_tulu_speech.py --video isl_recognition\demo_input\demo_video.mp4 --artifacts isl_recognition\transfer_pack --mapping isl_recognition\artifacts\tulu_sentence_map_kn.json --top-k 5 --speak
```
