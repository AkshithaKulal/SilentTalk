# Install PyTorch with CUDA 12.1 (for office GPU training).
# Run from repo root with venv activated:
#   .\scripts\install_torch_cuda.ps1

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (Test-Path ".\.venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }
elseif (Test-Path ".\silent-venv\Scripts\Activate.ps1") { . .\silent-venv\Scripts\Activate.ps1 }

Write-Host ""
Write-Host "=== PyTorch CUDA 12.1 (~2.5 GB download) ===" -ForegroundColor Cyan
python -c "import torch; print('before:', torch.__version__, 'cuda=', torch.cuda.is_available())"

pip install --upgrade pip
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 `
    --index-url https://download.pytorch.org/whl/cu121

Write-Host ""
python -c @"
import torch
ok = torch.cuda.is_available()
print('after:', torch.__version__, 'cuda=', ok)
if ok:
    print('gpu:', torch.cuda.get_device_name(0))
    print('vram GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
else:
    print('CUDA still False — update NVIDIA driver or check GPU.')
    exit(1)
"@

Write-Host ""
Write-Host "OK — run office_overnight.py again." -ForegroundColor Green
