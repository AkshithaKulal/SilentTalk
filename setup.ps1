# SilentTalk Setup Script - Run once after git clone: .\setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "        SilentTalk Setup Script           " -ForegroundColor Cyan
Write-Host "==========================================`n" -ForegroundColor Cyan

# 1. Python check
Write-Host "[1/7] Checking Python..." -ForegroundColor Yellow
try {
    $pyver = & python --version 2>&1
    Write-Host "      $pyver" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# 2. Create venv
Write-Host "[2/7] Creating virtual environment 'silent-venv'..." -ForegroundColor Yellow
if (-not (Test-Path "silent-venv")) {
    python -m venv silent-venv
    Write-Host "      Created silent-venv/" -ForegroundColor Green
} else {
    Write-Host "      silent-venv/ already exists, skipping." -ForegroundColor DarkGray
}
. .\silent-venv\Scripts\Activate.ps1
Write-Host "      Activated silent-venv" -ForegroundColor Green

# 3. Install dependencies
Write-Host "[3/7] Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

$packages = @("torch","transformers","peft","mediapipe","scikit-learn","joblib","opencv-python","pillow","scipy","numpy","flask","sounddevice","pyttsx3","gdown")
foreach ($pkg in $packages) {
    Write-Host "      Installing $pkg..." -NoNewline
    pip install $pkg --quiet
    Write-Host " done" -ForegroundColor Green
}

# IndicTransToolkit - needs C++ Build Tools, fall back to source install
Write-Host "      Installing IndicTransToolkit..." -NoNewline
pip install IndicTransToolkit --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host " C++ Build Tools missing - installing from source..." -ForegroundColor Yellow
    pip install sacrebleu nltk sacremoses indic-nlp-library --quiet
    if (-not (Test-Path "IndicTransToolkit_repo")) {
        git clone https://github.com/AI4Bharat/IndicTransToolkit.git IndicTransToolkit_repo --quiet
    }
    Push-Location IndicTransToolkit_repo
    pip install -e . --no-build-isolation --quiet 2>&1 | Out-Null
    Pop-Location
    Write-Host "      IndicTransToolkit installed from source" -ForegroundColor Green
} else {
    Write-Host " done" -ForegroundColor Green
}

# 4. Download checkpoint-1500 LoRA adapter from Google Drive
Write-Host "[4/7] Checking LoRA adapter (checkpoint-1500)..." -ForegroundColor Yellow
$ckptPath = "checkpoint-1500-inference"
if (Test-Path "$ckptPath\adapter_model.safetensors") {
    Write-Host "      [OK] LoRA checkpoint already present" -ForegroundColor Green
} else {
    Write-Host "      Downloading from Google Drive (~23MB)..."
    python -c "import gdown; gdown.download_folder('https://drive.google.com/drive/folders/1RgEDcwom1ny6IFfnSvFyfTNzYAeA4DPd', output='checkpoint-1500-inference', quiet=False, use_cookies=False); print('Done.')"
    if (Test-Path "$ckptPath\adapter_model.safetensors") {
        Write-Host "      [OK] checkpoint-1500 downloaded" -ForegroundColor Green
    } else {
        Write-Host "      [WARN] Download may have failed - check Drive permissions" -ForegroundColor Yellow
    }
}

# 5. Check ISL classifier artifacts
Write-Host "[5/7] Checking ISL classifier artifacts..." -ForegroundColor Yellow
$missing = @()
$artifacts = @{
    "sign_classifier.joblib" = "isl_recognition\transfer_pack\sign_classifier.joblib"
    "label_encoder.joblib"   = "isl_recognition\transfer_pack\label_encoder.joblib"
}
foreach ($name in $artifacts.Keys) {
    if (Test-Path $artifacts[$name]) {
        Write-Host "      [OK] $name" -ForegroundColor Green
    } else {
        Write-Host "      [MISSING] $name" -ForegroundColor Red
        $missing += $name
    }
}
Write-Host "      [INFO] Base 1B translation model downloads automatically on first run (~8.3GB)" -ForegroundColor DarkCyan
$mmsCachePath = "$env:USERPROFILE\.cache\huggingface\hub\models--facebook--mms-tts-kan"
if (Test-Path $mmsCachePath) {
    Write-Host "      [OK] MMS Kannada TTS cached" -ForegroundColor Green
} else {
    Write-Host "      [INFO] MMS TTS will download on first run (~140MB)" -ForegroundColor Yellow
}

# 6. Download MediaPipe models
Write-Host "[6/7] Downloading MediaPipe models..." -ForegroundColor Yellow
python -c "
import sys; sys.path.insert(0, 'isl_recognition')
from extract_landmarks import download_if_missing, HAND_MODEL_URL, POSE_MODEL_URL
from pathlib import Path
m = Path('isl_recognition/models'); m.mkdir(exist_ok=True)
print('  hand landmarker...', end=' ', flush=True)
download_if_missing(HAND_MODEL_URL, m / 'hand_landmarker.task')
print('done')
print('  pose landmarker...', end=' ', flush=True)
download_if_missing(POSE_MODEL_URL, m / 'pose_landmarker_lite.task')
print('done')
"

# 7. Download verification_set from Google Drive
Write-Host "[7/7] Downloading verification_set (sample sign videos)..." -ForegroundColor Yellow
$verificationPath = "isl_recognition\verification_set"
if (Test-Path $verificationPath) {
    $videoCount = (Get-ChildItem $verificationPath -Recurse -Include "*.MOV","*.mp4" -ErrorAction SilentlyContinue).Count
    Write-Host "      verification_set/ already present ($videoCount videos), skipping." -ForegroundColor DarkGray
} else {
    Write-Host "      Downloading from Google Drive..."
    python -c "import gdown; gdown.download_folder('https://drive.google.com/drive/folders/1Hia3uO4VBa-NI38CpBKjvE_GsWO6TXxP', output='isl_recognition/verification_set', quiet=False, use_cookies=False); print('Done.')"
    Write-Host "      Done." -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "  MISSING (copy from training machine):" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host "    - $m" -ForegroundColor Red }
}
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor White
Write-Host "    cd frontend" -ForegroundColor Green
Write-Host "    npm install" -ForegroundColor Green
Write-Host "    npm run build" -ForegroundColor Green
Write-Host "    cd .." -ForegroundColor Green
Write-Host "    . .\silent-venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "    python app.py" -ForegroundColor Green
Write-Host "    Open: http://localhost:5000" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
