#!/usr/bin/env python3
"""
SilentTalk — One-shot setup script.

Run this ONCE after cloning the repo on any new machine:
    python setup.py

What it does:
  1. Checks Python version (3.10+ required)
  2. Creates a virtual environment (.venv)
  3. Installs all Python dependencies (PyTorch CUDA, FastAPI, MediaPipe, etc.)
  4. Installs IndicTransToolkit from local repo
  5. Downloads MediaPipe model files (hand + pose)
  6. Downloads LoRA checkpoint if missing
  7. Checks/downloads MMS-TTS Kannada model
  8. Installs Node.js frontend dependencies
  9. Builds the React frontend
 10. Prints run instructions

Usage after setup:
    Windows:  .venv\\Scripts\\activate  then  python app.py
    Linux:    source .venv/bin/activate  then  python app.py
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

# ── colour helpers ────────────────────────────────────────────────────────────
def green(s):  return f"\033[92m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def cyan(s):   return f"\033[96m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"

sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None

ROOT    = Path(__file__).resolve().parent
IS_WIN  = platform.system() == "Windows"
VENV    = ROOT / "silent-venv"
PY_VENV = VENV / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python")
PIP     = [str(PY_VENV), "-m", "pip"]

def run(cmd, check=True, cwd=None, capture=False):
    kwargs = dict(cwd=cwd or ROOT, check=check)
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    return subprocess.run(cmd, **kwargs)

def section(title):
    print(f"\n{cyan('─' * 60)}")
    print(f"{bold(cyan(f'  {title}'))}")
    print(f"{cyan('─' * 60)}")

def ok(msg):   print(green(f"  ✓  {msg}"))
def warn(msg): print(yellow(f"  ⚠  {msg}"))
def info(msg): print(f"     {msg}")
def fail(msg): print(red(f"  ✗  {msg}")); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
print(bold(cyan("""
╔══════════════════════════════════════════════╗
║        SilentTalk Setup Script               ║
║  ISL → Kannada Speech  |  One-shot install   ║
╚══════════════════════════════════════════════╝
""")))

# ── Step 1: Python version ────────────────────────────────────────────────────
section("Step 1/9 — Python version check")
major, minor = sys.version_info[:2]
info(f"Python {major}.{minor}.{sys.version_info[2]}")
if major < 3 or minor < 10:
    fail(f"Python 3.10+ required. You have {major}.{minor}. Install from https://python.org")
ok(f"Python {major}.{minor} — OK")

# ── Step 2: Create venv ───────────────────────────────────────────────────────
section("Step 2/9 — Virtual environment")
if VENV.exists():
    ok(".venv already exists — skipping creation")
else:
    info("Creating .venv ...")
    run([sys.executable, "-m", "venv", str(VENV)])
    ok(".venv created")

if not PY_VENV.exists():
    # fallback: if .venv exists from previous setup, use it
    fallback = ROOT / ".venv" / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python")
    if fallback.exists():
        warn("silent-venv not found but .venv exists — using .venv")
        globals()['PY_VENV'] = fallback
        globals()['PIP'] = [str(fallback), "-m", "pip"]
    else:
        fail(f"Python not found in venv at {PY_VENV}")
ok(f"Venv Python: {PY_VENV}")

# ── Step 3: Upgrade pip ───────────────────────────────────────────────────────
section("Step 3/9 — Installing Python packages")
info("Upgrading pip...")
run([*PIP, "install", "--upgrade", "pip", "--quiet"])

# ── PyTorch with CUDA 12.1 ────────────────────────────────────────────────────
info("Checking PyTorch + CUDA...")
result = run([str(PY_VENV), "-c",
    "import torch; print(torch.cuda.is_available()); print(torch.__version__)"],
    check=False, capture=True)

if result.returncode == 0:
    out = result.stdout.strip().lower()
    has_cuda_build = "+cu" in out or "cuda" in out
    cuda_ok = out.startswith("true")
    if cuda_ok:
        ok(f"PyTorch + CUDA ready: {result.stdout.strip()}")
    elif has_cuda_build:
        info(f"PyTorch CUDA build installed but cuda.is_available()=False — check NVIDIA driver.\n  {result.stdout.strip()}")
    else:
        info("Installing PyTorch 2.5.1 with CUDA 12.1 (this is ~2.5GB, may take a few minutes)...")
        run([*PIP, "install",
             "torch==2.5.1+cu121",
             "torchvision==0.20.1+cu121",
             "torchaudio==2.5.1+cu121",
             "--index-url", "https://download.pytorch.org/whl/cu121",
             "--quiet"])
        ok("PyTorch + CUDA 12.1 installed")
else:
    info("Installing PyTorch 2.5.1 with CUDA 12.1 (this is ~2.5GB, may take a few minutes)...")
    run([*PIP, "install",
         "torch==2.5.1+cu121",
         "torchvision==0.20.1+cu121",
         "torchaudio==2.5.1+cu121",
         "--index-url", "https://download.pytorch.org/whl/cu121",
         "--quiet"])
    ok("PyTorch + CUDA 12.1 installed")

# ── Core packages ─────────────────────────────────────────────────────────────
req_file = ROOT / "requirements-app.txt"
if req_file.exists():
    info(f"Installing from {req_file.name}...")
    run([*PIP, "install", "-r", str(req_file), "--quiet"])
    ok("requirements-app.txt installed")
else:
    packages = [
        "fastapi==0.115.12",
        "uvicorn[standard]==0.34.3",
        "python-multipart==0.0.20",
        "opencv-python==4.10.0.84",
        "mediapipe>=0.10.30",
        "scikit-learn>=1.5.0",
        "joblib>=1.3.0",
        "numpy<2.0",
        "scipy>=1.11",
        "transformers==4.46.1",
        "huggingface_hub>=0.36",
        "peft==0.15.2",
        "datasets",
        "accelerate",
        "parler-tts==0.2.3",
        "gdown",
        "sacrebleu",
        "nltk",
        "sacremoses",
        "indic-nlp-library",
    ]
    info(f"Installing {len(packages)} packages...")
    for pkg in packages:
        name = pkg.split("==")[0].split(">=")[0].split("<")[0]
        print(f"     Installing {name}...", end=" ", flush=True)
        result = run([*PIP, "install", pkg, "--quiet"], check=False)
        if result.returncode == 0:
            print(green("done"))
        else:
            print(yellow("warn — may already be installed"))
    ok("Core packages done")

# ── IndicTransToolkit ─────────────────────────────────────────────────────────
info("Installing IndicTransToolkit from local repo...")
toolkit_path = ROOT / "IndicTrans2" / "huggingface_interface" / "IndicTransToolkit_repo"
if toolkit_path.exists():
    result = run([*PIP, "install", "-e", str(toolkit_path),
                  "--no-build-isolation", "--quiet"], check=False)
    if result.returncode == 0:
        ok("IndicTransToolkit installed from local repo")
    else:
        warn("IndicTransToolkit local install failed — trying pip...")
        run([*PIP, "install", "IndicTransToolkit", "--quiet"], check=False)
else:
    warn("Local IndicTransToolkit repo not found — installing from pip")
    run([*PIP, "install", "IndicTransToolkit", "--quiet"], check=False)

# ── Step 4: MediaPipe models ──────────────────────────────────────────────────
section("Step 4/9 — MediaPipe model files")
models_dir = ROOT / "isl_recognition" / "models"
models_dir.mkdir(parents=True, exist_ok=True)

HAND_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

for url, filename in [(HAND_URL, "hand_landmarker.task"),
                      (POSE_URL, "pose_landmarker_lite.task")]:
    dest = models_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        ok(f"{filename} already present")
    else:
        info(f"Downloading {filename}...")
        import urllib.request
        urllib.request.urlretrieve(url, dest)
        ok(f"{filename} downloaded")

# ── Step 5: LoRA checkpoint ───────────────────────────────────────────────────
section("Step 5/9 — LoRA checkpoint (checkpoint-1500-inference)")
ckpt = ROOT / "checkpoint-1500-inference" / "adapter_model.safetensors"
if ckpt.exists():
    ok("LoRA checkpoint already present")
else:
    warn("LoRA checkpoint missing — downloading from Google Drive (~23MB)...")
    try:
        run([str(PY_VENV), "-c",
             "import gdown; gdown.download_folder("
             "'https://drive.google.com/drive/folders/1RgEDcwom1ny6IFfnSvFyfTNzYAeA4DPd',"
             " output='checkpoint-1500-inference', quiet=False, use_cookies=False)"])
        ok("LoRA checkpoint downloaded")
    except Exception as e:
        warn(f"Auto-download failed: {e}")
        warn("Manually place checkpoint-1500-inference/ at repo root.")
        warn("Drive link: https://drive.google.com/drive/folders/1RgEDcwom1ny6IFfnSvFyfTNzYAeA4DPd")

# ── Step 6: Sign model artifacts ──────────────────────────────────────────────
section("Step 6/9 — ISL sign model")
bilstm = ROOT / "isl_recognition" / "transfer_pack" / "sign_bilstm.pt"
clf_path = ROOT / "isl_recognition" / "transfer_pack" / "sign_classifier.joblib"
if bilstm.exists():
    ok("sign_bilstm.pt present (BiLSTM — live app default)")
elif clf_path.exists():
    ok("sign_classifier.joblib present (legacy MLP fallback)")
else:
    warn("No sign model in isl_recognition/transfer_pack/")
    warn("Pull latest from git or train: python isl_recognition/train_sequence.py")

# ── Step 7: HuggingFace .env ──────────────────────────────────────────────────
section("Step 7/9 — Environment (.env)")
env_file = ROOT / ".env"
template = (
    "HF_TOKEN=your_huggingface_token_here\n"
    "SARVAM_API_KEY=your_sarvam_api_key_here\n"
)
if env_file.exists():
    ok(".env file already present")
    text = env_file.read_text(encoding="utf-8")
    if "SARVAM_API_KEY" not in text:
        env_file.write_text(text.rstrip() + "\nSARVAM_API_KEY=your_sarvam_api_key_here\n", encoding="utf-8")
        warn("Added SARVAM_API_KEY line to .env — fill in for Bulbul TTS (optional)")
else:
    warn(".env file missing — creating template")
    env_file.write_text(template, encoding="utf-8")
    warn("Edit .env: HF_TOKEN (required for first model download)")
    warn("Optional: SARVAM_API_KEY for best Kannada TTS (https://dashboard.sarvam.ai)")

# ── Step 8: Node.js frontend ──────────────────────────────────────────────────
section("Step 8/9 — Frontend (Node.js + React build)")
frontend_dir = ROOT / "frontend"
node_modules = frontend_dir / "node_modules"

# Check node/npm
npm_result = run(["npm", "--version"], check=False, capture=True)
if npm_result.returncode != 0:
    warn("npm not found! Install Node.js 18+ from https://nodejs.org")
    warn("Then re-run: cd frontend && npm install && npm run build")
else:
    ok(f"npm {npm_result.stdout.strip()} found")
    if node_modules.exists():
        ok("node_modules already installed")
    else:
        info("Running npm install...")
        run(["npm", "install"], cwd=str(frontend_dir))
        ok("npm install done")

    info("Building React frontend...")
    result = run(["npm", "run", "build"], cwd=str(frontend_dir), check=False)
    if result.returncode == 0:
        ok("React build complete → static/react/")
    else:
        warn("Frontend build failed. Run manually: cd frontend && npm run build")

# ── Step 9: Final check ───────────────────────────────────────────────────────
section("Step 9/9 — Final verification")

checks = {
    "sign model (BiLSTM or MLP)": (
        ROOT / "isl_recognition/transfer_pack/sign_bilstm.pt"
        if (ROOT / "isl_recognition/transfer_pack/sign_bilstm.pt").exists()
        else ROOT / "isl_recognition/transfer_pack/sign_classifier.joblib"
    ),
    "hand_landmarker.task":      ROOT / "isl_recognition/models/hand_landmarker.task",
    "pose_landmarker.task":      ROOT / "isl_recognition/models/pose_landmarker_lite.task",
    "LoRA checkpoint":           ROOT / "checkpoint-1500-inference/adapter_model.safetensors",
    "React build":               ROOT / "static/react/index.html",
    ".env file":                 ROOT / ".env",
}

all_ok = True
for name, path in checks.items():
    if path.exists():
        ok(name)
    else:
        warn(f"MISSING: {name}")
        all_ok = False

# ── Done ──────────────────────────────────────────────────────────────────────
print()
if all_ok:
    print(bold(green("""
╔══════════════════════════════════════════════════════════╗
║  ✓  Setup Complete — Everything is ready!               ║
╠══════════════════════════════════════════════════════════╣
║  Run (production — one port):                            ║
║    Windows:  .\\silent-venv\\Scripts\\activate          ║
║              python app.py                                 ║
║    Open:     http://localhost:5000                       ║
║                                                          ║
║  Phone camera on Wi-Fi:                                    ║
║              python app.py --https                       ║
║    Open:     https://YOUR_LAN_IP:5000                    ║
║                                                          ║
║  Dev (UI hot-reload):                                    ║
║    Terminal 1: .\\scripts\\dev.ps1 backend               ║
║    Terminal 2: .\\scripts\\dev.ps1 frontend              ║
║    Open:       http://localhost:5173                     ║
║                                                          ║
║  First run downloads ~4–8 GB models from HuggingFace.    ║
║  Edit .env: HF_TOKEN + optional SARVAM_API_KEY.        ║
╚══════════════════════════════════════════════════════════╝
""")))
else:
    print(bold(yellow("""
╔══════════════════════════════════════════════════════════╗
║  ⚠  Setup mostly complete — some items need attention   ║
║  Check warnings above and fix before running app.py      ║
╚══════════════════════════════════════════════════════════╝
""")))
