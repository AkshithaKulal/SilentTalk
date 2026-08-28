# ============================================================
#  SilentTalk — Train on office machine, push model to GitHub
#  Usage:
#    .\train_and_push.ps1
#  Optional — use more INCLUDE data:
#    .\train_and_push.ps1 -IncludeRoot "F:\Include_dataset\extracted"
#  Optional — run full extraction first:
#    .\train_and_push.ps1 -IncludeRoot "F:\Include_dataset\extracted" -Extract
# ============================================================
param(
    [string]$IncludeRoot   = "",          # path to INCLUDE extracted videos
    [switch]$Extract       = $false,      # re-run landmark extraction
    [string]$VenvPath      = ".venv",     # venv folder
    [string]$Branch        = "main",      # git branch to push to
    [int]   $MinPerClass   = 5            # minimum samples per class
)

$ErrorActionPreference = "Stop"

# ── helpers ─────────────────────────────────────────────────────────────────
function Log($msg) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $msg" -ForegroundColor Cyan }
function Ok($msg)  { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $msg" -ForegroundColor Green }
function Err($msg) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ERROR: $msg" -ForegroundColor Red }

# ── locate python ────────────────────────────────────────────────────────────
$PY = "$VenvPath\Scripts\python.exe"
if (-not (Test-Path $PY)) {
    # fallback to any venv name
    foreach ($v in @("silent-venv", ".venv", "venv")) {
        if (Test-Path "$v\Scripts\python.exe") { $PY = "$v\Scripts\python.exe"; break }
    }
}
if (-not (Test-Path $PY)) {
    Err "No venv found. Tried .venv, silent-venv, venv. Create one first."
    exit 1
}
Ok "Using Python: $PY"

# ── check git is clean (warn only) ───────────────────────────────────────────
Log "Checking git status..."
$gitStatus = git status --porcelain 2>&1
if ($gitStatus) {
    Log "Uncommitted changes found — will commit model artifacts on top."
}

# ── optionally run landmark extraction ──────────────────────────────────────
if ($Extract -and $IncludeRoot) {
    if (-not (Test-Path $IncludeRoot)) {
        Err "IncludeRoot not found: $IncludeRoot"
        exit 1
    }
    Log "Running landmark extraction from: $IncludeRoot"
    Log "This may take 30-120 minutes depending on dataset size..."
    & $PY isl_recognition\extract_landmarks.py `
        --input  $IncludeRoot `
        --output isl_recognition\landmarks `
        --skip-existing
    if ($LASTEXITCODE -ne 0) { Err "Extraction failed."; exit 1 }
    Ok "Extraction done."
} elseif ($Extract -and -not $IncludeRoot) {
    Err "-Extract requires -IncludeRoot path. Example: -IncludeRoot F:\Include_dataset\extracted"
    exit 1
}

# ── check we have landmarks ───────────────────────────────────────────────────
$npy = (Get-ChildItem "isl_recognition\landmarks" -Filter "*.npy" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($npy -eq 0) {
    Err "No landmark .npy files found in isl_recognition\landmarks\."
    Err "Run with -Extract -IncludeRoot <path> to extract first."
    exit 1
}
Log "Found $npy landmark files."

# ── train ─────────────────────────────────────────────────────────────────────
Log "Starting training (min-per-class=$MinPerClass)..."
Log "Output: isl_recognition\transfer_pack\"
Log "This takes 2-10 minutes..."

& $PY isl_recognition\train_classifier.py `
    --landmarks  isl_recognition\landmarks `
    --out-dir    isl_recognition\transfer_pack `
    --min-per-class $MinPerClass

if ($LASTEXITCODE -ne 0) { Err "Training failed."; exit 1 }
Ok "Training complete."

# ── verify artifacts exist ────────────────────────────────────────────────────
$clf = "isl_recognition\transfer_pack\sign_classifier.joblib"
$le  = "isl_recognition\transfer_pack\label_encoder.joblib"
$rep = "isl_recognition\transfer_pack\train_report.json"

foreach ($f in @($clf, $le, $rep)) {
    if (-not (Test-Path $f)) { Err "Missing expected artifact: $f"; exit 1 }
}

# Print accuracy from report
$report = Get-Content $rep | ConvertFrom-Json
Ok "Model accuracy:"
Ok "  Top-1 : $([math]::Round($report.accuracy * 100, 1))%"
Ok "  Top-3 : $([math]::Round($report.top3 * 100, 1))%"
Ok "  Top-5 : $([math]::Round($report.top5 * 100, 1))%"
Ok "  Classes: $($report.num_classes)   Feature dim: $($report.feature_dim)"

# ── git: stage artifacts ──────────────────────────────────────────────────────
Log "Staging model artifacts for git..."

# Force-add joblib files (they might be in .gitignore from older rules)
git add -f $clf
git add -f $le
git add -f $rep

# Also stage the updated train_classifier.py and app.py
git add isl_recognition\train_classifier.py
git add app.py

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$acc1  = [math]::Round($report.accuracy * 100, 1)
$acc3  = [math]::Round($report.top3 * 100, 1)
$nc    = $report.num_classes
$fd    = $report.feature_dim

$commitMsg = "model: retrain with richer features top1=$acc1% top3=$acc3% classes=$nc feat_dim=$fd [$timestamp]"

Log "Committing: $commitMsg"
git commit -m $commitMsg

if ($LASTEXITCODE -ne 0) {
    Log "Nothing new to commit — artifacts unchanged."
} else {
    Ok "Committed."
}

# ── git push ──────────────────────────────────────────────────────────────────
Log "Pushing to origin/$Branch ..."
git push origin $Branch

if ($LASTEXITCODE -ne 0) {
    Err "Push failed. Check your git remote and credentials."
    Err "You can push manually: git push origin $Branch"
    exit 1
}

Ok "======================================================"
Ok "  DONE. Model trained and pushed to GitHub."
Ok "  On your laptop, run:"
Ok "    git pull"
Ok "  Then restart Flask:"
Ok "    python app.py"
Ok "======================================================"
