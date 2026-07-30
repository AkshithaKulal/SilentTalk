# PowerShell setup for IndicTrans2 LoRA (8GB GPU)
# Run from: C:\Sarastra\SilentTalk
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "==> Cloning IndicTrans2 (if needed)..."
if (-not (Test-Path "IndicTrans2")) {
    git clone https://github.com/AI4Bharat/IndicTrans2
}
else {
    Write-Host "IndicTrans2 already exists - skipping clone"
}

Set-Location "IndicTrans2\huggingface_interface"

Write-Host "==> Running install.sh..."
$gitBash = "C:\Program Files\Git\bin\bash.exe"
if (Test-Path $gitBash) {
    & $gitBash -lc "source install.sh"
}
else {
    bash -lc "source install.sh"
}

if ((Test-Path "IndicTransToolkit") -and -not (Test-Path "IndicTransToolkit_repo")) {
    Rename-Item "IndicTransToolkit" "IndicTransToolkit_repo"
}

Write-Host "==> Installing Python packages..."
python -m pip install -e IndicTransToolkit_repo -q
# Pin torchao: latest breaks torch 2.5 / Windows; 0.9.0 works with transformers 4.53.2
python -m pip install "torchao==0.9.0" -q
python -m pip install sacrebleu pandas datasets huggingface_hub accelerate sentencepiece nltk -q
python -m pip install "transformers==4.53.2" "peft==0.15.2" -q

Write-Host "==> Copying helper scripts..."
Copy-Item "$Root\lora_local\patch_files.py" . -Force
Copy-Item "$Root\lora_local\download_model.py" . -Force
Copy-Item "$Root\lora_local\prepare_data.py" . -Force
Copy-Item "$Root\lora_local\run_train.ps1" . -Force
Copy-Item "$Root\lora_local\resume_train.ps1" . -Force
Copy-Item "$Root\lora_local\run_train.sh" . -Force
Copy-Item "$Root\lora_local\resume_train.sh" . -Force

Write-Host "==> Verifying import..."
python -c "from IndicTransToolkit import IndicProcessor, IndicDataCollator; print('Import successful')"

Write-Host ""
Write-Host "Env setup done."
Write-Host "Next:"
Write-Host "  cd C:\Sarastra\SilentTalk\IndicTrans2\huggingface_interface"
Write-Host "  python patch_files.py"
Write-Host "  # edit download_model.py - paste YOUR_HF_TOKEN"
Write-Host "  python download_model.py"
Write-Host "  python prepare_data.py"
Write-Host "  .\run_train.ps1"
