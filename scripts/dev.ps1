# SilentTalk — two services for local development
#
# Service 1: Backend API + ML models  →  http://localhost:5000
# Service 2: Frontend (Vite + HMR)    →  http://localhost:5173
#
# Usage (from repo root):
#   .\scripts\dev.ps1 backend    # run in terminal 1
#   .\scripts\dev.ps1 frontend   # run in terminal 2
#   Open http://localhost:5173  (NOT 5000 — that is API only in dev)

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("backend", "frontend", "build")]
    [string]$Target
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

switch ($Target) {
    "backend" {
        Write-Host ""
        Write-Host "=== SilentTalk backend (API + models) ===" -ForegroundColor Cyan
        Write-Host "Port 5000 — sign predict, translate, TTS" -ForegroundColor Gray
        Write-Host "Open UI at http://localhost:5173 (run: .\scripts\dev.ps1 frontend)" -ForegroundColor Yellow
        Write-Host ""
        if (Test-Path ".\.venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }
        elseif (Test-Path ".\silent-venv\Scripts\Activate.ps1") { . .\silent-venv\Scripts\Activate.ps1 }
        python app.py
    }
    "frontend" {
        Write-Host ""
        Write-Host "=== SilentTalk frontend (Vite) ===" -ForegroundColor Cyan
        Write-Host "Port 5173 — proxies /api to localhost:5000" -ForegroundColor Gray
        Write-Host "Start backend first: .\scripts\dev.ps1 backend" -ForegroundColor Yellow
        Write-Host ""
        Set-Location frontend
        npm run dev
    }
    "build" {
        Write-Host "Building frontend → static/react (for single-port demo on :5000)" -ForegroundColor Cyan
        Set-Location frontend
        npm run build
        Write-Host "Done. Run: python app.py  then open http://localhost:5000" -ForegroundColor Green
    }
}
