# SilentTalk — one-shot setup after git clone (Windows)
# Run from repo root:  .\setup.ps1
#
# This wraps setup.py (Python 3.10+, Node.js 18+, Git, NVIDIA GPU recommended).

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

Write-Host ""
Write-Host "SilentTalk setup — calling setup.py" -ForegroundColor Cyan
Write-Host "Requires: Python 3.10+, Node.js 18+, Git" -ForegroundColor Gray
Write-Host ""

python --version 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

python setup.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Phone camera (HTTPS on LAN):" -ForegroundColor Yellow
Write-Host "  . .\silent-venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "  python app.py --https" -ForegroundColor Green
Write-Host "  Open https://YOUR_LAN_IP:5000 on phone (accept certificate warning)" -ForegroundColor Green
