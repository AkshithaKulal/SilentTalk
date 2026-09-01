# SilentTalk — one-shot setup after git clone (Windows)
# Run from repo root:  .\setup.ps1
#
# Installs Python deps, Node.js (winget), MSVC Build Tools (winget), and builds the UI.
# For auto system installs, run PowerShell **as Administrator** on a fresh machine.

param(
    [switch]$SkipSystemDeps
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

Write-Host ""
Write-Host "SilentTalk setup — calling setup.py" -ForegroundColor Cyan
Write-Host "Requires: Python 3.10+, Git (Node.js + MSVC auto-installed via winget)" -ForegroundColor Gray
if (-not $isAdmin) {
    Write-Host "Tip: Run as Administrator for automatic Node.js + MSVC install on first setup" -ForegroundColor Yellow
}
Write-Host ""

python --version 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

$setupArgs = @()
if ($SkipSystemDeps) { $setupArgs += "--skip-system-deps" }

python setup.py @setupArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Phone camera (HTTPS on LAN):" -ForegroundColor Yellow
Write-Host "  . .\silent-venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "  python app.py --https" -ForegroundColor Green
Write-Host "  Open https://YOUR_LAN_IP:5000 on phone (accept certificate warning)" -ForegroundColor Green
