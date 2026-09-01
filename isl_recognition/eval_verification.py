#!/usr/bin/env python3
"""Evaluate sign_bilstm.pt on verification_set videos (same path as live app).

Download sample data (once):
  python -m pip install gdown
  python isl_recognition/eval_verification.py --download

Run accuracy test:
  python isl_recognition/eval_verification.py

From repo root with venv active.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ISL = ROOT / "isl_recognition"
sys.path.insert(0, str(ISL))

MODELS = ISL / "models"
ARTIFACTS = ISL / "transfer_pack"
VSET = ISL / "verification_set"
MODEL_PATH = ARTIFACTS / "sign_bilstm.pt"
REPORT_PATH = ARTIFACTS / "sequence_train_report.json"

VERIFY_DRIVE = "https://drive.google.com/drive/folders/1Hia3uO4VBa-NI38CpBKjvE_GsWO6TXxP"


def label_from_folder(folder: Path) -> str:
    return re.sub(r"^\d+\.\s*", "", folder.name).strip()


def download_verification_set() -> None:
    try:
        import gdown
    except ImportError:
        print("Install gdown: python -m pip install gdown")
        sys.exit(1)
    VSET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading verification_set to {VSET} ...")
    gdown.download_folder(
        VERIFY_DRIVE,
        output=str(VSET),
        quiet=False,
        use_cookies=False,
    )
    n = len(list(VSET.rglob("*.mp4")) + list(VSET.rglob("*.mov")) + list(VSET.rglob("*.MOV")))
    print(f"Done — {n} video files under verification_set/")


def load_model():
    import torch
    from sequence_model import load_bundle, predict_topk

    if not MODEL_PATH.exists():
        print(f"Missing model: {MODEL_PATH}")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, device = load_bundle(MODEL_PATH, device)
    print(f"Loaded {MODEL_PATH.name} — {len(classes)} classes on {device}")
    return model, classes, device, predict_topk


def iter_clips(root: Path):
    if not root.exists():
        return
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        label = label_from_folder(folder)
        vids = sorted(
            v for v in folder.iterdir()
            if v.suffix.lower() in (".mp4", ".mov", ".avi")
        )
        if vids:
            yield label, vids[0]


def run_eval(limit: int | None = None) -> int:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from extract_landmarks import (
        download_if_missing,
        extract_video,
        HAND_MODEL_URL,
        POSE_MODEL_URL,
    )

    clips = list(iter_clips(VSET))
    if not clips:
        print(f"No videos in {VSET}")
        print("Run: python isl_recognition/eval_verification.py --download")
        return 1

    if limit:
        clips = clips[:limit]

    model, classes, device, predict_topk = load_model()

    hand_model = download_if_missing(HAND_MODEL_URL, MODELS / "hand_landmarker.task")
    pose_model = download_if_missing(POSE_MODEL_URL, MODELS / "pose_landmarker_lite.task")
    BaseOptions = mp_python.BaseOptions
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    hand_opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    correct = top3_ok = top5_ok = total = 0
    results = []
    ts_offset = 0

    print(f"\nEvaluating {len(clips)} clips from verification_set\n")

    with vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm, \
         vision.HandLandmarker.create_from_options(hand_opts) as hand_lm:

        for label, video in clips:
            print(f"  {label:<24}", end="  ", flush=True)
            t0 = time.perf_counter()
            try:
                feats, _meta, ts_offset = extract_video(
                    video, pose_lm, hand_lm, frame_stride=2, timestamp_offset_ms=ts_offset,
                )
            except Exception as exc:
                print(f"ERROR: {exc}")
                total += 1
                results.append({"label": label, "pred": "ERROR", "ok": False})
                continue

            elapsed = int((time.perf_counter() - t0) * 1000)

            if feats.shape[0] == 0:
                print("NO FEATURES")
                total += 1
                results.append({"label": label, "pred": "NO_FEATURES", "ok": False})
                continue

            preds = predict_topk(model, classes, feats, device, k=5)
            top_labels = [p[0] for p in preds]
            top1, conf = preds[0]
            conf_pct = conf * 100

            ok = top1.lower() == label.lower()
            ok3 = label.lower() in [x.lower() for x in top_labels[:3]]
            ok5 = label.lower() in [x.lower() for x in top_labels[:5]]
            correct += int(ok)
            top3_ok += int(ok3)
            top5_ok += int(ok5)
            total += 1

            mark = "OK" if ok else ("TOP3" if ok3 else f"-> {top1}")
            print(f"{conf_pct:5.1f}%  {mark}  [{elapsed}ms]")
            results.append({
                "label": label,
                "pred": top1,
                "conf": conf_pct,
                "top5": preds,
                "ok": ok,
                "ok3": ok3,
            })

    print("\n" + "=" * 72)
    print(f"  VERIFICATION SET — BiLSTM ({MODEL_PATH.name})")
    print("=" * 72)
    if REPORT_PATH.exists():
        import json
        r = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        print(f"  Training report (landmarks, official test): "
              f"top1={r.get('test_acc', 0)*100:.1f}%  "
              f"top3={r.get('test_top3', 0)*100:.1f}%  "
              f"({r.get('version', '?')})")
        print("  (Video eval below is harder — extraction + single clip per sign)")
    print("-" * 72)
    if total:
        print(f"  Top-1 : {correct}/{total} = {100*correct/total:.1f}%")
        print(f"  Top-3 : {top3_ok}/{total} = {100*top3_ok/total:.1f}%")
        print(f"  Top-5 : {top5_ok}/{total} = {100*top5_ok/total:.1f}%")
    print("=" * 72)

    wrong = [r for r in results if not r.get("ok")]
    if wrong:
        print("\nMisclassified:")
        for r in wrong:
            t3 = ", ".join(f"{p[0]} {p[1]*100:.0f}%" for p in r.get("top5", [])[:3])
            print(f"  {r['label']:<22} -> {r['pred']} ({r.get('conf', 0):.0f}%)   top3: {t3}")

    return 0 if total else 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM on verification_set videos")
    parser.add_argument("--download", action="store_true", help="Download verification_set from Google Drive")
    parser.add_argument("--limit", type=int, default=None, help="Max clips to test")
    args = parser.parse_args()
    if args.download:
        download_verification_set()
        return
    sys.exit(run_eval(limit=args.limit))


if __name__ == "__main__":
    main()
