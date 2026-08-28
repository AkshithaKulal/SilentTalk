# ============================================================
#  SilentTalk — Train on office machine, push model to GitHub
#
#  Usage (existing landmarks):
#    .\train_and_push.ps1
#
#  Usage (extract from INCLUDE first):
#    .\train_and_push.ps1 -IncludeRoot "F:\Include_dataset\extracted" -Extract
# ============================================================
param(
    [string]$IncludeRoot   = "",
    [switch]$Extract       = $false,
    [string]$VenvPath      = ".venv",
    [string]$Branch        = "main",
    [int]   $MinPerClass   = 5,
    [string]$GithubUser    = "AkshithaKulal"   # GitHub account to push as
)

$ErrorActionPreference = "Stop"

function Log($m)  { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] WARN: $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ERROR: $m" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  ================================================" -ForegroundColor Magenta
Write-Host "    SilentTalk — Train + Push Script" -ForegroundColor Magenta
Write-Host "  ================================================" -ForegroundColor Magenta
Write-Host ""

# ════════════════════════════════════════════════════════════
#  STEP 1 — PULL LATEST CODE FROM GITHUB FIRST
# ════════════════════════════════════════════════════════════
Log "STEP 1/6 — Pulling latest code from GitHub..."

$remoteUrl = git remote get-url origin 2>&1
if ($LASTEXITCODE -ne 0) { Fail "No git remote 'origin'. Run: git remote add origin https://github.com/AkshithaKulal/SilentTalk.git" }
Log "Remote: $remoteUrl"

# Make sure remote URL uses the right account
if ($remoteUrl -notmatch "AkshithaKulal") {
    Log "Fixing remote URL to use AkshithaKulal account..."
    git remote set-url origin "https://AkshithaKulal@github.com/AkshithaKulal/SilentTalk.git"
    Ok "Remote URL updated."
}

git fetch origin 2>&1 | Out-Null
git pull origin $Branch 2>&1
if ($LASTEXITCODE -ne 0) {
    Warn "git pull had issues — checking if it is just 'already up to date'..."
}
Ok "Code is up to date."

# ════════════════════════════════════════════════════════════
#  STEP 2 — CHECK & SWITCH GH CLI ACCOUNT (before training)
#           Fail fast here — not after hours of training
# ════════════════════════════════════════════════════════════
Log "STEP 2/6 — Checking GitHub credentials..."

# Check if gh CLI is available
$ghExists = Get-Command gh -ErrorAction SilentlyContinue
if ($ghExists) {
    Log "GitHub CLI found. Checking active account..."

    $ghStatus = gh auth status 2>&1
    $activeUser = ($ghStatus | Select-String "Active account: true" -Context 1,0).Context.PreContext `
                  | Select-String "Logged in to github.com account (\S+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }

    # Simpler parse — find active account line
    $activeLine = $ghStatus | Select-String "Active account: true"
    # The account name is two lines above
    $statusLines = $ghStatus -split "`n"
    $activeAccount = ""
    for ($i = 0; $i -lt $statusLines.Count; $i++) {
        if ($statusLines[$i] -match "Active account: true" -and $i -ge 1) {
            if ($statusLines[$i-1] -match "account (\S+)") {
                $activeAccount = $Matches[1]
            }
        }
    }

    if ($activeAccount) {
        Log "Currently active gh account: $activeAccount"
    }

    if ($activeAccount -ne $GithubUser) {
        Log "Switching gh CLI to $GithubUser..."
        $switchOut = gh auth switch --user $GithubUser 2>&1
        Log "$switchOut"

        # Verify switch worked
        $verifyStatus = gh auth status 2>&1
        if ($verifyStatus -notmatch "$GithubUser.*Active account: true" -and
            ($verifyStatus | Select-String "Active account: true" | Select-String $GithubUser).Count -eq 0) {

            # Try alternate check
            $activeNow = gh api user --jq .login 2>&1
            if ($activeNow -ne $GithubUser) {
                Write-Host ""
                Write-Host "  Cannot switch to $GithubUser automatically." -ForegroundColor Red
                Write-Host "  $GithubUser may not be logged in on this machine." -ForegroundColor Red
                Write-Host ""
                Write-Host "  Run this first, then re-run this script:" -ForegroundColor Yellow
                Write-Host "    gh auth login --hostname github.com --git-protocol https" -ForegroundColor White
                Write-Host "  (Choose: Login with browser or paste token)" -ForegroundColor White
                Write-Host ""
                Fail "Authentication setup required. See above."
            }
        }
        Ok "Switched to $GithubUser."
    } else {
        Ok "Already active as $GithubUser."
    }
} else {
    Warn "GitHub CLI (gh) not found. Falling back to git credential manager."
    Warn "If push fails with 403, install gh CLI: https://cli.github.com"
}

# Now do an actual dry-run push to confirm access
Log "Testing push access (dry-run — nothing is sent)..."
$dryRun = git push --dry-run origin $Branch 2>&1
if ($LASTEXITCODE -ne 0) {
    $dryStr = $dryRun -join " "
    Write-Host ""
    Write-Host "  PUSH TEST FAILED:" -ForegroundColor Red
    $dryRun | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "  Fix: make sure $GithubUser has push access to origin" -ForegroundColor Yellow
    Fail "Push access denied. Aborting before training."
}
Ok "Push access confirmed. Safe to train."

# ════════════════════════════════════════════════════════════
#  STEP 3 — LOCATE PYTHON VENV
# ════════════════════════════════════════════════════════════
Log "STEP 3/6 — Locating Python venv..."

$PY = ""
foreach ($v in @($VenvPath, ".venv", "venv", "silent-venv")) {
    if (Test-Path "$v\Scripts\python.exe") { $PY = "$v\Scripts\python.exe"; break }
}
if (-not $PY) { Fail "No venv found. Tried: $VenvPath, .venv, venv, silent-venv.`nRun setup.ps1 first." }

$pyVer = & $PY --version 2>&1
Ok "Python: $pyVer  ($PY)"

$pkgCheck = & $PY -c "import mediapipe, sklearn, joblib, numpy; print('ok')" 2>&1
if ($pkgCheck -notmatch "ok") {
    Fail "Missing Python packages. Run: $PY -m pip install -r isl_recognition\requirements.txt"
}
Ok "All required packages present."

# ════════════════════════════════════════════════════════════
#  STEP 4 — LANDMARK EXTRACTION (optional)
# ════════════════════════════════════════════════════════════
Log "STEP 4/6 — Landmarks..."

if ($Extract) {
    if (-not $IncludeRoot) { Fail "-Extract requires -IncludeRoot. Example: -IncludeRoot F:\Include_dataset\extracted" }
    if (-not (Test-Path $IncludeRoot)) { Fail "IncludeRoot not found: $IncludeRoot" }

    $vidCount = (Get-ChildItem $IncludeRoot -Recurse -Include "*.mp4","*.mov","*.avi","*.mkv" -ErrorAction SilentlyContinue | Measure-Object).Count
    Log "Found $vidCount videos in $IncludeRoot"
    Log "Extracting landmarks (this may take 30-120 mins for full dataset)..."

    & $PY isl_recognition\extract_landmarks.py `
        --input  $IncludeRoot `
        --output isl_recognition\landmarks `
        --skip-existing
    if ($LASTEXITCODE -ne 0) { Fail "Extraction failed." }
    Ok "Extraction done."
} else {
    Log "Skipping extraction (no -Extract flag). Using existing landmarks."
}

$npy = (Get-ChildItem "isl_recognition\landmarks" -Filter "*.npy" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($npy -eq 0) { Fail "No .npy files found in isl_recognition\landmarks.`nRun with: -Extract -IncludeRoot <path>" }
Ok "Found $npy landmark files."

# ════════════════════════════════════════════════════════════
#  STEP 5 — TRAIN
# ════════════════════════════════════════════════════════════
Log "STEP 5/6 — Training classifier (min-per-class=$MinPerClass)..."
Log "Writing artifacts to isl_recognition\transfer_pack\"
Log "Estimated time: 5-20 mins depending on dataset size..."

& $PY isl_recognition\train_classifier.py `
    --landmarks   isl_recognition\landmarks `
    --out-dir     isl_recognition\transfer_pack `
    --min-per-class $MinPerClass

if ($LASTEXITCODE -ne 0) { Fail "Training failed." }

# Verify all artifacts exist
foreach ($f in @(
    "isl_recognition\transfer_pack\sign_classifier.joblib",
    "isl_recognition\transfer_pack\label_encoder.joblib",
    "isl_recognition\transfer_pack\train_report.json"
)) {
    if (-not (Test-Path $f)) { Fail "Expected artifact missing: $f" }
}

$rep  = Get-Content "isl_recognition\transfer_pack\train_report.json" | ConvertFrom-Json
$acc1 = [math]::Round($rep.accuracy * 100, 1)
$acc3 = [math]::Round($rep.top3 * 100, 1)
$acc5 = [math]::Round($rep.top5 * 100, 1)
$nc   = $rep.num_classes
$fd   = $rep.feature_dim
$ns   = $rep.samples_used

Ok "Training complete."
Ok "  Top-1  : $acc1%"
Ok "  Top-3  : $acc3%"
Ok "  Top-5  : $acc5%"
Ok "  Classes: $nc     Samples: $ns     Feature dim: $fd"

if ($rep.accuracy -lt 0.70) {
    Warn "Top-1 $acc1% is below 70% — check your data quality."
}

# ════════════════════════════════════════════════════════════
#  STEP 6 — COMMIT + PUSH
# ════════════════════════════════════════════════════════════
Log "STEP 6/6 — Committing and pushing to GitHub..."

git add -f "isl_recognition\transfer_pack\sign_classifier.joblib"
git add -f "isl_recognition\transfer_pack\label_encoder.joblib"
git add -f "isl_recognition\transfer_pack\train_report.json"
git add -f "isl_recognition\transfer_pack\classification_report.txt"
git add    "isl_recognition\train_classifier.py"
git add    "app.py"
git add    ".gitignore"

$ts  = Get-Date -Format "yyyy-MM-dd HH:mm"
$msg = "model: top1=$acc1% top3=$acc3% classes=$nc samples=$ns feat_dim=$fd [$ts]"
Log "Commit message: $msg"

git commit -m $msg 2>&1
if ($LASTEXITCODE -ne 0) {
    Warn "Nothing new to commit (model identical to last push). Skipping."
} else {
    Ok "Committed."
}

Log "Pushing to origin/$Branch ..."
git push origin $Branch 2>&1
if ($LASTEXITCODE -ne 0) { Fail "Push failed. Run manually: git push origin $Branch" }

# ════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "  ================================================" -ForegroundColor Green
Write-Host "   DONE — Model pushed to GitHub" -ForegroundColor Green
Write-Host "   Top-1: $acc1%   Top-3: $acc3%   Classes: $nc" -ForegroundColor Green
Write-Host "  ================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Back on your laptop run:" -ForegroundColor White
Write-Host "    git pull" -ForegroundColor Yellow
Write-Host "    python app.py" -ForegroundColor Yellow
Write-Host "  ================================================" -ForegroundColor Green
Write-Host ""
