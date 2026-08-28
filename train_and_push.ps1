# ============================================================
#  SilentTalk — Train on office machine, push model to GitHub
#
#  Usage:
#    .\train_and_push.ps1
#
#  With full INCLUDE extraction first:
#    .\train_and_push.ps1 -IncludeRoot "F:\Include_dataset\extracted" -Extract
#
#  With existing landmarks (skip extraction):
#    .\train_and_push.ps1
# ============================================================
param(
    [string]$IncludeRoot = "",
    [switch]$Extract     = $false,
    [string]$VenvPath    = ".venv",
    [string]$Branch      = "main",
    [int]   $MinPerClass = 5
)

$ErrorActionPreference = "Stop"

# ── colour helpers ────────────────────────────────────────────────────────────
function Log($m)  { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] WARN: $m" -ForegroundColor Yellow }
function Err($m)  { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ERROR: $m" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  ================================================" -ForegroundColor Magenta
Write-Host "    SilentTalk — Train + Push Script" -ForegroundColor Magenta
Write-Host "  ================================================" -ForegroundColor Magenta
Write-Host ""

# ════════════════════════════════════════════════════════════
#  STEP 1 — GIT CREDENTIAL CHECK  (before anything else)
#           Fail fast here, not after hours of training
# ════════════════════════════════════════════════════════════
Log "STEP 1/5 — Checking git credentials and remote..."

# Get remote URL
$remoteUrl = git remote get-url origin 2>&1
if ($LASTEXITCODE -ne 0) { Err "No git remote 'origin' found. Run: git remote add origin <url>" }
Log "Remote: $remoteUrl"

# Extract expected owner from remote URL
# handles both https://github.com/Owner/Repo.git and git@github.com:Owner/Repo.git
if ($remoteUrl -match "github\.com[:/]([^/]+)/") {
    $expectedOwner = $Matches[1]
    Log "Expected GitHub owner: $expectedOwner"
} else {
    Warn "Could not parse owner from remote URL. Proceeding anyway."
    $expectedOwner = ""
}

# Test push access with a dry-run (does not actually push anything)
Log "Testing push access (dry-run)..."
$dryRun = git push --dry-run origin $Branch 2>&1
$dryRunStr = $dryRun -join " "

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Push test FAILED. Output:" -ForegroundColor Red
    $dryRun | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "  Fix options:" -ForegroundColor Yellow
    Write-Host "  1. Switch to correct GitHub account:" -ForegroundColor Yellow
    Write-Host "     git remote set-url origin https://AkshithaKulal@github.com/AkshithaKulal/SilentTalk.git" -ForegroundColor White
    Write-Host "     git push  (enter your PAT as password)" -ForegroundColor White
    Write-Host ""
    Write-Host "  2. Create a PAT at: https://github.com/settings/tokens" -ForegroundColor Yellow
    Write-Host "     Scope needed: repo (full control)" -ForegroundColor White
    Write-Host ""
    Write-Host "  3. Or use GitHub CLI:" -ForegroundColor Yellow
    Write-Host "     gh auth login" -ForegroundColor White
    Write-Host ""
    Err "Aborting — fix credentials BEFORE training to avoid wasting time."
}

Ok "Push access confirmed for origin/$Branch"

# Verify we are pushing as the right user
$whoami = git config user.email 2>&1
Log "Git identity: $whoami"
if ($expectedOwner -and $whoami -notmatch $expectedOwner -and $whoami -notmatch "akshitha" -and $whoami -notmatch "kulal") {
    Warn "Git user '$whoami' may not match repo owner '$expectedOwner'."
    Warn "If push fails with 403, run: git config user.email 'your@email.com'"
}

# ════════════════════════════════════════════════════════════
#  STEP 2 — LOCATE PYTHON VENV
# ════════════════════════════════════════════════════════════
Log "STEP 2/5 — Locating Python venv..."

$PY = ""
foreach ($v in @($VenvPath, ".venv", "venv", "silent-venv")) {
    if (Test-Path "$v\Scripts\python.exe") { $PY = "$v\Scripts\python.exe"; break }
}
if (-not $PY) { Err "No venv found. Tried: $VenvPath, .venv, venv, silent-venv" }

$pyVer = & $PY --version 2>&1
Ok "Python: $pyVer  ($PY)"

# Quick sanity — check required packages
$pkgCheck = & $PY -c "import mediapipe, sklearn, joblib, numpy; print('ok')" 2>&1
if ($pkgCheck -notmatch "ok") { Err "Missing packages. Run: pip install -r isl_recognition\requirements.txt" }
Ok "Required packages present."

# ════════════════════════════════════════════════════════════
#  STEP 3 — LANDMARK EXTRACTION (optional)
# ════════════════════════════════════════════════════════════
Log "STEP 3/5 — Landmarks..."

if ($Extract) {
    if (-not $IncludeRoot) { Err "-Extract requires -IncludeRoot. Example: -IncludeRoot F:\Include_dataset\extracted" }
    if (-not (Test-Path $IncludeRoot)) { Err "IncludeRoot not found: $IncludeRoot" }

    Log "Extracting landmarks from: $IncludeRoot"
    Log "This may take 30-120 minutes on full dataset..."
    & $PY isl_recognition\extract_landmarks.py `
        --input  $IncludeRoot `
        --output isl_recognition\landmarks `
        --skip-existing
    if ($LASTEXITCODE -ne 0) { Err "Extraction failed." }
    Ok "Extraction done."
} else {
    Log "Skipping extraction (-Extract not set). Using existing landmarks."
}

$npy = (Get-ChildItem "isl_recognition\landmarks" -Filter "*.npy" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($npy -eq 0) { Err "No .npy files in isl_recognition\landmarks. Run with -Extract -IncludeRoot <path>." }
Ok "Found $npy landmark files."

# ════════════════════════════════════════════════════════════
#  STEP 4 — TRAIN
# ════════════════════════════════════════════════════════════
Log "STEP 4/5 — Training classifier (min-per-class=$MinPerClass)..."
Log "Output -> isl_recognition\transfer_pack\"
Log "Estimated time: 3-15 minutes..."

& $PY isl_recognition\train_classifier.py `
    --landmarks   isl_recognition\landmarks `
    --out-dir     isl_recognition\transfer_pack `
    --min-per-class $MinPerClass

if ($LASTEXITCODE -ne 0) { Err "Training failed." }

# Verify artifacts
foreach ($f in @(
    "isl_recognition\transfer_pack\sign_classifier.joblib",
    "isl_recognition\transfer_pack\label_encoder.joblib",
    "isl_recognition\transfer_pack\train_report.json"
)) {
    if (-not (Test-Path $f)) { Err "Expected artifact missing after training: $f" }
}

$rep    = Get-Content "isl_recognition\transfer_pack\train_report.json" | ConvertFrom-Json
$acc1   = [math]::Round($rep.accuracy * 100, 1)
$acc3   = [math]::Round($rep.top3 * 100, 1)
$acc5   = [math]::Round($rep.top5 * 100, 1)
$nc     = $rep.num_classes
$fd     = $rep.feature_dim

Ok "Training complete."
Ok "  Top-1 : $acc1%   Top-3 : $acc3%   Top-5 : $acc5%"
Ok "  Classes: $nc     Feature dim: $fd"

# Quality gate — warn if accuracy dropped
if ($rep.accuracy -lt 0.70) {
    Warn "Top-1 accuracy $acc1% is below 70% baseline. Check your data."
    Warn "Pushing anyway — review after pull."
}

# ════════════════════════════════════════════════════════════
#  STEP 5 — COMMIT + PUSH
# ════════════════════════════════════════════════════════════
Log "STEP 5/5 — Committing and pushing..."

git add -f "isl_recognition\transfer_pack\sign_classifier.joblib"
git add -f "isl_recognition\transfer_pack\label_encoder.joblib"
git add -f "isl_recognition\transfer_pack\train_report.json"
git add -f "isl_recognition\transfer_pack\classification_report.txt"
git add    "isl_recognition\train_classifier.py"
git add    "app.py"
git add    ".gitignore"

$ts  = Get-Date -Format "yyyy-MM-dd HH:mm"
$msg = "model: top1=$acc1% top3=$acc3% classes=$nc feat_dim=$fd samples=$($rep.samples_used) [$ts]"
Log "Commit: $msg"

git commit -m $msg 2>&1
if ($LASTEXITCODE -ne 0) {
    Warn "git commit returned non-zero (possibly nothing new to commit). Checking..."
    $pending = git status --porcelain 2>&1
    if (-not $pending) {
        Warn "Nothing to commit — model unchanged from last push."
    }
}

Log "Pushing to origin/$Branch ..."
git push origin $Branch 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Err "Push failed! Run manually: git push origin $Branch"
}

# ════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "  ================================================" -ForegroundColor Green
Write-Host "   DONE — Model trained and pushed to GitHub" -ForegroundColor Green
Write-Host "  ================================================" -ForegroundColor Green
Write-Host "   Top-1: $acc1%   Top-3: $acc3%   Classes: $nc" -ForegroundColor Green
Write-Host ""
Write-Host "   On your laptop run:" -ForegroundColor White
Write-Host "     git pull" -ForegroundColor Yellow
Write-Host "     python app.py" -ForegroundColor Yellow
Write-Host "  ================================================" -ForegroundColor Green
Write-Host ""
