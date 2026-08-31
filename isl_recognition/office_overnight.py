#!/usr/bin/env python3
"""Office overnight: landmarks → NEW BiLSTM train → push only small model files.

Run this from the SilentTalk repo AFTER include_prepare.py has finished
(or pass --run-prepare if it has not). Leave it running when you go home.
It will not push until training succeeds. It never adds videos or landmarks.

  cd <SilentTalk>
  .\\.venv\\Scripts\\Activate.ps1    # or silent-venv, whatever the office uses
  python .\\isl_recognition\\office_overnight.py --root F:\\include_dataset

At home:  git pull   then start the app. It loads sign_bilstm.pt if present.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ISL = Path(__file__).resolve().parent
sys.path.insert(0, str(ISL))
from torch_device import cuda_ready, gpu_summary, resolve_device

PACK = ISL / "transfer_pack"
LOG = ISL / "artifacts" / "office_overnight.log"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}
PUSH_FILES = [
    PACK / "sign_bilstm.pt",
    PACK / "sign_bilstm.classes.json",
    PACK / "sequence_train_report.json",
]
MAX_PT_MB = 80


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd: list[str], cwd: Path | None = None) -> int:
    log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd or REPO))
    if proc.returncode != 0:
        log(f"FAILED exit={proc.returncode}")
    return proc.returncode


def count_videos(extracted: Path) -> int:
    if not extracted.is_dir():
        return 0
    return sum(1 for p in extracted.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def git_ok_to_push() -> bool:
    name = subprocess.check_output(["git", "config", "user.name"], cwd=REPO, text=True).strip()
    email = subprocess.check_output(["git", "config", "user.email"], cwd=REPO, text=True).strip()
    log(f"git identity: {name} <{email}>")
    # Fail NOW (before hours of work) if this machine cannot push.
    dry = subprocess.run(
        ["git", "push", "--dry-run", "origin", "main"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if dry.returncode != 0:
        log("git push --dry-run failed. Log into GitHub as AkshithaKulal on this PC first.")
        log(dry.stderr.strip() or dry.stdout.strip())
        return False
    log("git push --dry-run OK")
    return True


def push_models() -> int:
    missing = [p for p in PUSH_FILES if not p.exists()]
    if missing:
        log("missing model files: " + ", ".join(str(p) for p in missing))
        return 1
    pt = PACK / "sign_bilstm.pt"
    mb = pt.stat().st_size / (1024 * 1024)
    log(f"sign_bilstm.pt is {mb:.1f} MB")
    if mb > MAX_PT_MB:
        log(f"refusing to commit: file larger than {MAX_PT_MB} MB")
        return 1

    rels = [str(p.relative_to(REPO)) for p in PUSH_FILES]
    if run(["git", "add", "--", *rels]) != 0:
        return 1
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], cwd=REPO, text=True
    ).strip()
    if not staged:
        log("nothing new to commit (model already on this commit)")
        return run(["git", "push", "origin", "main"])

    msg = "Add INCLUDE sequence model from office overnight train."
    if run(["git", "commit", "-m", msg]) != 0:
        return 1
    if run(["git", "pull", "--rebase", "origin", "main"]) != 0:
        log("rebase failed — model is on disk but not pushed. Fix conflicts tomorrow.")
        return 1
    return run(["git", "push", "origin", "main"])


def check_gpu(require: bool) -> bool:
    log(gpu_summary())
    if cuda_ready():
        try:
            import torch

            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            log(f"GPU OK: {name} ({mem:.1f} GB VRAM) — training will use CUDA")
        except Exception as exc:  # noqa: BLE001
            log(f"GPU detected but query failed: {exc}")
        return True

    msg = (
        "CUDA not available — training would run on CPU (very slow).\n"
        "  Fix: .\\scripts\\install_torch_cuda.ps1\n"
        "  Or:  python setup.py\n"
        '  Verify: python -c "import torch; print(torch.cuda.is_available())"'
    )
    if require:
        log("ERROR: " + msg.replace("\n", "\n  "))
        return False
    log("WARNING: " + msg.replace("\n", "\n  "))
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"F:\include_dataset"))
    parser.add_argument(
        "--run-prepare",
        action="store_true",
        help="Run include_prepare.py first (skip if it already finished in another window)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Train even if extract looks like the old 5-category slice",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU training if CUDA is missing (default: require GPU)",
    )
    parser.add_argument(
        "--skip-landmarks",
        action="store_true",
        help="Skip landmark extraction (use existing .npy files)",
    )
    args = parser.parse_args()
    require_gpu = not args.allow_cpu

    extracted = args.root / "extracted"
    py = sys.executable
    log("=" * 72)
    log(f"office overnight start  python={py}")
    log(f"repo={REPO}")
    log(f"dataset={args.root}")

    if not git_ok_to_push():
        return 1

    if not check_gpu(require_gpu):
        return 1

    try:
        resolve_device("cuda" if cuda_ready() else "auto", require_gpu=require_gpu)
    except SystemExit:
        return 1

    if args.run_prepare:
        prep = ISL / "include_official" / "include_prepare.py"
        if run([py, str(prep), "--root", str(args.root)]) != 0:
            log("prepare failed — not training, not pushing")
            return 1

    nvid = count_videos(extracted)
    log(f"extracted videos: {nvid}")
    if nvid < 2500 and not args.allow_partial:
        log(
            "Still looks incomplete (want ~4292). "
            "Wait for include_prepare to finish, then re-run this script. "
            "Or pass --allow-partial to train on whatever is extracted."
        )
        return 1

    land = ISL / "landmarks"
    if not args.skip_landmarks:
        if run([
            py,
            str(ISL / "extract_landmarks.py"),
            "--input",
            str(extracted),
            "--output",
            str(land),
            "--skip-existing",
        ]) != 0:
            log("landmark extract failed — not pushing")
            return 1
    else:
        log("skipping landmark extraction (--skip-landmarks)")

    train_cmd = [
        py,
        str(ISL / "train_sequence.py"),
        "--landmarks",
        str(land),
        "--out",
        str(PACK),
        "--epochs",
        str(args.epochs),
        "--device",
        "cuda" if cuda_ready() else "auto",
    ]
    if require_gpu:
        train_cmd.append("--require-gpu")

    if run(train_cmd) != 0:
        log("training failed — not pushing")
        return 1

    log("training OK — pushing model files only (no videos, no landmarks)")
    rc = push_models()
    if rc == 0:
        log("PUSHED. At home: git pull  then start SilentTalk.")
    else:
        log("push failed. Model is on this office disk under isl_recognition/transfer_pack/")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
