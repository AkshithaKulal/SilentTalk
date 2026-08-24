#!/usr/bin/env pwsh
# SilentTalk Setup Script
# Creates 'silent-venv', installs all dependencies, verifies model artifacts.
# Run once after cloning: .\setup.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         SilentTalk Setup Script          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝`n" -ForegroundColor Cyan

# ── 1. Python check ──────────────────────────────────────────────────────────
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
$pyver = python --version 2>&1
if ($LASTEXITCODE -ne 0) { Write-Error "Python not found. Install Python 3.10+ from python.org" }
Write-Host "      $pyver" -ForegroundColor Green

# ── 2. Create venv ───────────────────────────────────────────────────────────
Write-Host "[2/6] Creating virtual environment 'silent-venv'..." -ForegroundColor Yellow
if (-not (Test-Path "silent-venv")) {
    python -m venv silent-venv
    Write-Host "      Created silent-venv/" -ForegroundColor Green
} else {
    Write-Host "      silent-venv/ already exists, skipping." -ForegroundColor DarkGray
}

# Activate
. .\silent-venv\Scripts\Activate.ps1
Write-Host "      Activated silent-venv" -ForegroundColor Green

# ── 3. Upgrade pip + install dependencies ────────────────────────────────────
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

$packages = @(
    "torch",
    "transformers",
    "peft",
    "IndicTransToolkit",
    "mediapipe",
    "scikit-learn",
    "joblib",
    "opencv-python",
    "pillow",
    "scipy",
    "numpy",
    "flask",
    "sounddevice",
    "pyttsx3",
    "gdown"
)

foreach ($pkg in $packages) {
    Write-Host "      Installing $pkg..." -NoNewline
    pip install $pkg --quiet
    Write-Host " done" -ForegroundColor Green
}

# ── 4. Verify model artifacts ────────────────────────────────────────────────
Write-Host "[4/6] Checking model artifacts..." -ForegroundColor Yellow

$artifacts = @{
    "ISL Classifier"         = "isl_recognition\transfer_pack\sign_classifier.joblib"
    "Label Encoder"          = "isl_recognition\transfer_pack\label_encoder.joblib"
    "IndicTrans2 1B Base"    = "IndicTrans2\huggingface_interface\model_cache\indictrans2-en-indic-1B"
    "LoRA checkpoint-1500"   = "IndicTrans2\huggingface_interface\stage1_output\checkpoint-1500\adapter_model.safetensors"
}

$missing = @()
foreach ($name in $artifacts.Keys) {
    $path = $artifacts[$name]
    if (Test-Path $path) {
        Write-Host "      ✓ $name" -ForegroundColor Green
    } else {
        Write-Host "      ✗ $name — MISSING at $path" -ForegroundColor Red
        $missing += $name
    }
}

# Check HF cache for MMS TTS
$mmsCachePath = "$env:USERPROFILE\.cache\huggingface\hub\models--facebook--mms-tts-kan"
if (Test-Path $mmsCachePath) {
    Write-Host "      ✓ MMS Kannada TTS (HF cache)" -ForegroundColor Green
} else {
    Write-Host "      ✗ MMS TTS not cached — will download on first run (~140MB)" -ForegroundColor Yellow
}

# ── 5. Download MediaPipe models ─────────────────────────────────────────────
Write-Host "[5/7] Checking MediaPipe models..." -ForegroundColor Yellow
python -c "
import sys
sys.path.insert(0, 'isl_recognition')
from extract_landmarks import download_if_missing, HAND_MODEL_URL, POSE_MODEL_URL
from pathlib import Path
m = Path('isl_recognition/models')
m.mkdir(exist_ok=True)
print('  Downloading hand landmarker...')
download_if_missing(HAND_MODEL_URL, m / 'hand_landmarker.task')
print('  Downloading pose landmarker...')
download_if_missing(POSE_MODEL_URL, m / 'pose_landmarker_lite.task')
print('  MediaPipe models ready.')
"

# ── 6. Download verification_set from Google Drive ───────────────────────────
Write-Host "[6/7] Downloading verification_set (sample sign videos)..." -ForegroundColor Yellow
$verificationPath = "isl_recognition\verification_set"
if (Test-Path $verificationPath) {
    $videoCount = (Get-ChildItem $verificationPath -Recurse -Include "*.MOV","*.mp4" -ErrorAction SilentlyContinue).Count
    Write-Host "      verification_set/ already exists ($videoCount videos), skipping." -ForegroundColor DarkGray
} else {
    Write-Host "      Downloading from Google Drive (this may take a few minutes)..."
    python -c "
import gdown
gdown.download_folder(
    'https://drive.google.com/drive/folders/1Hia3uO4VBa-NI38CpBKjvE_GsWO6TXxP',
    output='isl_recognition/verification_set',
    quiet=False,
    use_cookies=False
)
print('  verification_set downloaded.')
"
    Write-Host "      Done." -ForegroundColor Green
}

# ── 7. Summary ───────────────────────────────────────────────────────────────
Write-Host "[7/7] Setup complete." -ForegroundColor Yellow

if ($missing.Count -gt 0) {
    Write-Host "`n⚠  Missing artifacts (copy from training machine):" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host "   - $m" -ForegroundColor Red }
    Write-Host ""
}

Write-Host "══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " To run the demo:" -ForegroundColor White
Write-Host "   . .\silent-venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "   python app.py" -ForegroundColor Green
Write-Host "   Then open http://localhost:5000" -ForegroundColor Green
Write-Host " To run CLI prediction:" -ForegroundColor White
Write-Host "   python isl_recognition\predict_tulu_speech.py --video <clip.MOV> --artifacts isl_recognition\transfer_pack --speak" -ForegroundColor Green
Write-Host "══════════════════════════════════════════════" -ForegroundColor Cyan
