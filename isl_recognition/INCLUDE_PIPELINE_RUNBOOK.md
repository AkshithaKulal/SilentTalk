# Production Runbook: INCLUDE Preprocessing and ISL Training

This is the production SOP for preprocessing and training.
Use this for repeatable runs, model promotion, and handover.
If you are only testing quickly, you may skip production controls, but do not use those outputs as release artifacts.

## 1) Scope

1. Audits INCLUDE videos for count and label coverage.
2. Extracts MediaPipe landmarks from each video.
3. Trains an ISL gloss classifier (MLP on pooled temporal features).
4. Produces versioned artifacts for downstream inference.

## 2) Repository and data prerequisites

1. Repo cloned locally.
2. INCLUDE extracted videos available, usually at `F:\Include_dataset\extracted`.
3. Python virtual environment in repo root (`.venv`).

Recommended minimum machine profile for full extraction + training:

1. CPU: 8 logical cores or better
2. RAM: 16 GB or more
3. Free disk: 80 GB or more (videos + landmarks + artifacts)

Expected INCLUDE layout (example):

`F:\Include_dataset\extracted\Greetings_1of2\Greetings\66. Sunday\clip.mov`

## 3) Setup (run from repo root)

```powershell
cd F:\SilentTalk
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r .\isl_recognition\requirements.txt
```

If `.venv` already exists, skip the venv creation line.

## 4) Production run contract (mandatory)

Define a run identifier and keep all run outputs traceable.

```powershell
$env:RUN_ID = Get-Date -Format "yyyyMMdd_HHmmss"
"RUN_ID=$env:RUN_ID"
```

For every production run, record:

1. dataset root path
2. git commit hash
3. python version
4. installed package lock

```powershell
git rev-parse HEAD
python -V
python -m pip freeze > .\isl_recognition\artifacts\pip_freeze_$env:RUN_ID.txt
```

## 5) Verify dataset root

```powershell
Test-Path F:\Include_dataset\extracted
Get-ChildItem F:\Include_dataset\extracted -Recurse -File |
  Where-Object { $_.Extension -match '^\.(mp4|avi|mov|mkv|webm|mpg|mpeg)$' } |
  Select-Object -First 20 FullName
```

If your root differs, use that path in all commands below.

## 6) Audit INCLUDE dataset (mandatory)

```powershell
python .\isl_recognition\audit_include_dataset.py --root F:\Include_dataset\extracted --min-per-class 5 --check-open-limit 500
```

Output report:

`isl_recognition\artifacts\include_audit_report.json`

Quick summary:

```powershell
$r = Get-Content .\isl_recognition\artifacts\include_audit_report.json | ConvertFrom-Json
"videos=$($r.total_video_files) labels=$($r.unique_label_count) low_labels=$($r.labels_below_min_per_class_count) open_fail=$($r.open_check.failed)"
```

## 7) Smoke extraction (first 20 videos)

```powershell
python .\isl_recognition\extract_landmarks.py --input F:\Include_dataset\extracted --output .\isl_recognition\landmarks --limit 20 --skip-existing
python .\isl_recognition\summarize_landmarks.py --landmarks .\isl_recognition\landmarks
```

If this fails, fix before full extraction.

## 8) Full extraction (resumable)

```powershell
python .\isl_recognition\extract_landmarks.py --input F:\Include_dataset\extracted --output .\isl_recognition\landmarks --skip-existing
```

Count check:

```powershell
$npy  = (Get-ChildItem .\isl_recognition\landmarks -Filter *.npy  -File | Measure-Object).Count
$json = (Get-ChildItem .\isl_recognition\landmarks -Filter *.json -File | Measure-Object).Count
"NPY=$npy JSON=$json"
```

## 9) Optional vocab coverage check

```powershell
python .\isl_recognition\check_vocab_coverage.py --landmarks .\isl_recognition\landmarks
```

Output:

`isl_recognition\artifacts\vocab_coverage.json`

## 10) Train classifier

```powershell
python .\isl_recognition\train_classifier.py --landmarks .\isl_recognition\landmarks --out-dir .\isl_recognition\artifacts --min-per-class 5
```

Expected output files:

1. `isl_recognition\artifacts\sign_classifier.joblib`
2. `isl_recognition\artifacts\label_encoder.joblib`
3. `isl_recognition\artifacts\train_report.json`
4. `isl_recognition\artifacts\classification_report.txt`

## 11) Production quality gates (required for release)

Do not promote a model unless all gates below pass.

Gate A: Dataset integrity

1. `total_video_files > 0`
2. `open_check.failed = 0` in audit sample (or documented exceptions)
3. Landmark counts match: `NPY == JSON`

Gate B: Label sufficiency

1. Any label below `min-per-class` is either:
   1. excluded deliberately, or
   2. recollected and rerun

Gate C: Training output completeness

1. `sign_classifier.joblib` exists
2. `label_encoder.joblib` exists
3. `train_report.json` exists
4. `classification_report.txt` exists

Gate D: Baseline quality threshold

Start with these baseline thresholds (adjust after team review):

1. accuracy >= 0.70
2. top3 >= 0.85
3. top5 >= 0.90

Quick check:

```powershell
$t = Get-Content .\isl_recognition\artifacts\train_report.json | ConvertFrom-Json
"acc=$($t.accuracy) top3=$($t.top3) top5=$($t.top5)"
```

## 12) Release packaging (training stage)

Package only required inference files:

1. `isl_recognition\artifacts\sign_classifier.joblib`
2. `isl_recognition\artifacts\label_encoder.joblib`
3. `isl_recognition\artifacts\tulu_sentence_map_kn.json`
4. `isl_recognition\artifacts\train_report.json`
5. `isl_recognition\artifacts\classification_report.txt`
6. `isl_recognition\artifacts\include_audit_report.json`

Optional archive:

```powershell
$pack = @(
   ".\isl_recognition\artifacts\sign_classifier.joblib",
   ".\isl_recognition\artifacts\label_encoder.joblib",
   ".\isl_recognition\artifacts\tulu_sentence_map_kn.json",
   ".\isl_recognition\artifacts\train_report.json",
   ".\isl_recognition\artifacts\classification_report.txt",
   ".\isl_recognition\artifacts\include_audit_report.json"
)
Compress-Archive -Path $pack -DestinationPath ".\isl_recognition\artifacts\isl_model_pack_$env:RUN_ID.zip" -Force
```

## 13) Run prediction (classifier only)

```powershell
python .\isl_recognition\predict_sign.py --video "FULL_PATH_TO_TEST_VIDEO.mp4" --artifacts .\isl_recognition\artifacts --top-k 5
```

## 14) Run spoken Tulu demo

This stage maps predicted labels to Tulu sentences from:

`isl_recognition\artifacts\tulu_sentence_map_kn.json`

Prediction + speech:

```powershell
python .\isl_recognition\predict_tulu_speech.py --video "FULL_PATH_TO_TEST_VIDEO.mp4" --artifacts .\isl_recognition\artifacts --mapping .\isl_recognition\artifacts\tulu_sentence_map_kn.json --top-k 5 --speak
```

## 15) Home-laptop fallback mode (no classifier artifacts)

If `sign_classifier.joblib` and `label_encoder.joblib` are missing, you can still test TTS flow:

```powershell
python .\isl_recognition\predict_tulu_speech.py --video .\isl_recognition\demo_input\demo_video.mp4 --artifacts .\isl_recognition\artifacts --mapping .\isl_recognition\artifacts\tulu_sentence_map_kn.json --top-k 5 --speak --allow-missing-model
```

Or pure text-to-speech:

```powershell
python .\isl_recognition\predict_tulu_speech.py --mapping .\isl_recognition\artifacts\tulu_sentence_map_kn.json --demo-text "ನಮಸ್ಕಾರ" --speak
```

## 16) Minimal files needed for prediction on another machine

Copy these files from office machine to target machine:

1. `isl_recognition\artifacts\sign_classifier.joblib`
2. `isl_recognition\artifacts\label_encoder.joblib`
3. `isl_recognition\artifacts\tulu_sentence_map_kn.json`
4. One test video file

## 17) Common errors and fixes

1. `Activate.ps1 is not recognized`
   1. Use dot-source form: `. .\.venv\Scripts\Activate.ps1`

2. `can't open file 'predict_tulu_speech.py'`
   1. Use full relative path from repo root: `python .\isl_recognition\predict_tulu_speech.py ...`

3. `No module named cv2`
   1. Install requirements in venv: `python -m pip install -r .\isl_recognition\requirements.txt`

4. `missing classifier artifacts`
   1. Train first using section 10, or copy artifact files from office machine.

5. `video not found`
   1. Check exact path with `Test-Path` and pass the real path to `--video`.

## 18) What to commit vs not commit

1. Commit code and docs.
2. Do not commit large landmarks/models/checkpoints unless explicitly required.
3. Keep generated artifacts local or in shared storage (Drive/USB) as needed.
