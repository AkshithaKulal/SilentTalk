#!/usr/bin/env python3
"""
SilentTalk — One-shot setup script.

Run this ONCE after cloning the repo on any new machine:
    python setup.py          # Windows: auto-installs Node.js + MSVC via winget
    python setup.py --skip-system-deps   # skip winget (manual prereqs)

What it does:
  1. Checks Python version (3.10+ required)
  2. Installs system tools on Windows (Node.js LTS, MSVC Build Tools via winget)
  3. Creates virtual environment (silent-venv)
  4. Installs all Python dependencies (PyTorch CUDA, FastAPI, MediaPipe, etc.)
  5. Installs IndicTransToolkit from local repo
  6. Downloads MediaPipe model files (hand + pose)
  7. Downloads LoRA checkpoint if missing
  8. Checks sign model + .env
  9. Installs Node.js frontend dependencies + builds React UI
 10. Prints run instructions

Usage after setup:
    Windows:  silent-venv\\Scripts\\activate  then  python app.py
    Linux:    source silent-venv/bin/activate  then  python app.py
"""

import argparse
import sys
import os
import shutil
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

parser = argparse.ArgumentParser(description="SilentTalk one-shot setup")
parser.add_argument(
    "--skip-system-deps",
    action="store_true",
    help="Do not auto-install Node.js / MSVC Build Tools (Windows winget)",
)
CLI = parser.parse_args()
AUTO_SYSTEM = IS_WIN and not CLI.skip_system_deps

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

# ── System prerequisites (Windows winget) ─────────────────────────────────────
def refresh_windows_path() -> None:
    if not IS_WIN:
        return
    try:
        import winreg
    except ImportError:
        return
    chunks = [os.environ.get("PATH", "")]
    for hive, subkey in (
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, r"Environment"),
    ):
        try:
            with winreg.OpenKey(hive, subkey) as key:
                chunks.insert(0, winreg.QueryValueEx(key, "Path")[0])
        except OSError:
            pass
    node_dir = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs"
    if node_dir.is_dir():
        chunks.insert(0, str(node_dir))
    os.environ["PATH"] = os.pathsep.join(p for part in chunks for p in part.split(os.pathsep) if p)


def winget_available() -> bool:
    return shutil.which("winget") is not None


def run_winget(install_args: list[str]) -> subprocess.CompletedProcess:
    cmd = [
        "winget", "install", *install_args,
        "--accept-package-agreements", "--accept-source-agreements",
    ]
    return run(cmd, check=False, capture=True)


def find_npm() -> str | None:
    npm = shutil.which("npm")
    if npm:
        return npm
    if IS_WIN:
        for candidate in (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "npm.cmd",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "nodejs" / "npm.cmd",
        ):
            if candidate.is_file():
                return str(candidate)
    return None


def has_msvc() -> bool:
    if not IS_WIN:
        return bool(shutil.which("gcc") or shutil.which("clang"))
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return False
    r = run(
        [
            str(vswhere), "-latest", "-products", "*",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find", r"**\Hostx64\x64\cl.exe",
        ],
        check=False,
        capture=True,
    )
    return r.returncode == 0 and bool(r.stdout.strip())


def ensure_nodejs() -> str | None:
    npm = find_npm()
    if npm:
        return npm
    if not AUTO_SYSTEM:
        return None
    if not winget_available():
        warn("winget not found — install Node.js 18+ LTS from https://nodejs.org")
        return None
    info("Node.js not found — installing OpenJS.NodeJS.LTS via winget (~2 min, Admin/UAC may appear)...")
    result = run_winget(["-e", "--id", "OpenJS.NodeJS.LTS"])
    refresh_windows_path()
    npm = find_npm()
    if npm:
        ok("Node.js installed")
    elif result.returncode != 0:
        warn(f"winget Node.js install failed (exit {result.returncode})")
        if result.stderr:
            info(result.stderr.strip()[:400])
    return npm


def ensure_msvc() -> bool:
    if has_msvc():
        return True
    if not AUTO_SYSTEM:
        return False
    if not winget_available():
        warn("winget not found — install MSVC Build Tools manually")
        return False
    info("MSVC not found — installing VS 2022 Build Tools via winget (~3 GB, 10–20 min, Admin required)...")
    info("Workload: Desktop development with C++ (VCTools)")
    override = (
        "--wait --passive "
        "--add Microsoft.VisualStudio.Workload.VCTools "
        "--includeRecommended"
    )
    result = run_winget(["-e", "--id", "Microsoft.VisualStudio.2022.BuildTools", "--override", override])
    refresh_windows_path()
    if has_msvc():
        ok("Microsoft C++ Build Tools installed")
        return True
    warn(f"MSVC install may still be running or needs a PC restart (winget exit {result.returncode})")
    if result.stdout:
        info(result.stdout.strip()[:500])
    return False


def ensure_system_prerequisites() -> None:
    section("Step 2/10 — System prerequisites")
    if not IS_WIN:
        if not shutil.which("gcc") and not shutil.which("clang"):
            warn("C compiler not found — install build-essential (Linux) or Xcode CLT (macOS)")
        else:
            ok("C/C++ compiler available")
        npm = find_npm()
        if npm:
            ok(f"npm found: {npm}")
        else:
            warn("npm not found — install Node.js 18+ from https://nodejs.org")
        return

    if AUTO_SYSTEM:
        info("Auto-install enabled (winget). Use --skip-system-deps to disable.")
        if not winget_available():
            warn("winget missing — install Node.js + MSVC Build Tools manually, or update Windows App Installer")
    else:
        info("Auto-install disabled (--skip-system-deps)")

    npm = ensure_nodejs()
    if npm:
        ver = run([npm, "--version"], check=False, capture=True)
        if ver.returncode == 0:
            ok(f"npm {ver.stdout.strip()}")
    elif not AUTO_SYSTEM:
        warn("npm not found — install Node.js 18+ LTS from https://nodejs.org")

    if has_msvc():
        ok("Microsoft C++ Build Tools (cl.exe) found")
    elif ensure_msvc():
        pass
    elif AUTO_SYSTEM:
        warn("MSVC still missing after winget — re-run setup as Administrator or restart PC")
    else:
        warn("MSVC Build Tools missing — required to build IndicTransToolkit on Windows")

# ─────────────────────────────────────────────────────────────────────────────
print(bold(cyan("""
╔══════════════════════════════════════════════╗
║        SilentTalk Setup Script               ║
║  ISL → Kannada Speech  |  One-shot install   ║
╚══════════════════════════════════════════════╝
""")))

# ── Step 1: Python version ────────────────────────────────────────────────────
section("Step 1/10 — Python version check")
major, minor = sys.version_info[:2]
info(f"Python {major}.{minor}.{sys.version_info[2]}")
if major < 3 or minor < 10:
    fail(f"Python 3.10+ required. You have {major}.{minor}. Install from https://python.org")
ok(f"Python {major}.{minor} — OK")

ensure_system_prerequisites()

# ── Step 3: Create venv ───────────────────────────────────────────────────────
section("Step 3/10 — Virtual environment")
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

# ── Step 4: Upgrade pip ───────────────────────────────────────────────────────
section("Step 4/10 — Installing Python packages")
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

# ── IndicTransToolkit (needs Cython + MSVC on Windows) ───────────────────────
def install_indic_trans_toolkit() -> bool:
    """Return True if IndicTransToolkit imports successfully."""
    section("IndicTransToolkit (translation preprocessing)")
    verify_existing = run(
        [str(PY_VENV), "-c", "from IndicTransToolkit import IndicProcessor"],
        check=False,
        capture=True,
    )
    if verify_existing.returncode == 0:
        ok("IndicTransToolkit already installed")
        return True

    if IS_WIN and not has_msvc():
        info("MSVC required — attempting install via winget...")
        ensure_msvc()

    info("Installing build helpers (Cython, setuptools, wheel)...")
    run([*PIP, "install", "cython", "setuptools", "wheel", "--upgrade", "--quiet"], check=False)

    toolkit_path = ROOT / "IndicTrans2" / "huggingface_interface" / "IndicTransToolkit_repo"
    installed = False
    if toolkit_path.exists():
        info("Building IndicTransToolkit from local repo (may take 2–5 min)...")
        result = run(
            [*PIP, "install", "-e", str(toolkit_path), "--no-build-isolation"],
            check=False,
        )
        installed = result.returncode == 0
    if not installed:
        warn("Local build failed — trying PyPI package...")
        result = run([*PIP, "install", "IndicTransToolkit", "--quiet"], check=False)
        installed = result.returncode == 0

    verify = run(
        [str(PY_VENV), "-c", "from IndicTransToolkit import IndicProcessor; print('ok')"],
        check=False,
        capture=True,
    )
    if verify.returncode == 0:
        ok("IndicTransToolkit installed")
        return True

    fail_msg = "IndicTransToolkit is REQUIRED for Kannada translation."
    if IS_WIN:
        warn(fail_msg)
        if AUTO_SYSTEM:
            warn("Re-run setup as Administrator, or restart PC after MSVC install, then:")
            warn("  .\\setup.ps1")
        else:
            warn("Re-run without --skip-system-deps, or install MSVC Build Tools manually:")
            warn("  https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            warn("  Then: pip install -e IndicTrans2\\huggingface_interface\\IndicTransToolkit_repo --no-build-isolation")
    else:
        warn(fail_msg)
        warn("Install build-essential (Linux) or Xcode CLT (macOS), then re-run setup.py")
    return False


install_indic_trans_toolkit()

# ── Step 5: MediaPipe models ──────────────────────────────────────────────────
section("Step 5/10 — MediaPipe model files")
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

# ── Step 6: LoRA checkpoint ───────────────────────────────────────────────────
section("Step 6/10 — LoRA checkpoint (checkpoint-1500-inference)")
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

# ── Step 7: Sign model artifacts ──────────────────────────────────────────────
section("Step 7/10 — ISL sign model")
bilstm = ROOT / "isl_recognition" / "transfer_pack" / "sign_bilstm.pt"
clf_path = ROOT / "isl_recognition" / "transfer_pack" / "sign_classifier.joblib"
if bilstm.exists():
    ok("sign_bilstm.pt present (BiLSTM — live app default)")
elif clf_path.exists():
    ok("sign_classifier.joblib present (legacy MLP fallback)")
else:
    warn("No sign model in isl_recognition/transfer_pack/")
    warn("Pull latest from git or train: python isl_recognition/train_sequence.py")

# ── Step 8: HuggingFace .env ──────────────────────────────────────────────────
section("Step 8/10 — Environment (.env)")
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

# ── Step 9: Node.js frontend ──────────────────────────────────────────────────
section("Step 9/10 — Frontend (Node.js + React build)")
frontend_dir = ROOT / "frontend"
node_modules = frontend_dir / "node_modules"
react_built = ROOT / "static" / "react" / "index.html"

npm_path = find_npm()
if not npm_path and AUTO_SYSTEM:
    npm_path = ensure_nodejs()
if not npm_path:
    warn("npm not found — Node.js is not installed on this machine.")
    warn("Install Node.js 18+ LTS from https://nodejs.org (check 'Add to PATH').")
    warn("Then run:  cd frontend && npm install && npm run build")
    if react_built.exists():
        ok("static/react/ already present — you can run python app.py without npm for now")
    else:
        warn("static/react/ missing — UI will not load until you build the frontend")
else:
    ok(f"npm found: {npm_path}")
    npm_result = run([npm_path, "--version"], check=False, capture=True)
    if npm_result.returncode == 0:
        ok(f"npm version {npm_result.stdout.strip()}")
    if node_modules.exists():
        ok("node_modules already installed")
    else:
        info("Running npm install...")
        run([npm_path, "install"], cwd=str(frontend_dir), check=False)

    if react_built.exists():
        ok("static/react/index.html already present")
        info("Skipping npm run build (delete static/react to force rebuild)")
    else:
        info("Building React frontend...")
        result = run([npm_path, "run", "build"], cwd=str(frontend_dir), check=False)
        if result.returncode == 0:
            ok("React build complete → static/react/")
        else:
            warn("Frontend build failed. Run manually: cd frontend && npm run build")

# ── Step 10: Final check ───────────────────────────────────────────────────────
section("Step 10/10 — Final verification")

checks = {
    "IndicTransToolkit": None,  # verified below
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
    if path is None:
        if name == "IndicTransToolkit":
            v = run(
                [str(PY_VENV), "-c", "from IndicTransToolkit import IndicProcessor"],
                check=False,
            )
            if v.returncode == 0:
                ok(name)
            else:
                warn(f"MISSING: {name} (translation will fail)")
                all_ok = False
        continue
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
