#!/usr/bin/env python3
"""Predict ISL gloss for one video using the trained landmark classifier.

Example:
  python predict_sign.py --video path\\to\\clip.MOV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

# Reuse extraction helpers from the sibling module
from extract_landmarks import (
    FEAT_DIM,
    HAND_MODEL_URL,
    POSE_MODEL_URL,
    download_if_missing,
    extract_video,
)
from train_classifier import sequence_to_features

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


def predict_video_topk(
    video: Path,
    artifacts: Path,
    models_dir: Path,
    top_k: int = 5,
    frame_stride: int = 2,
) -> tuple[list[tuple[str, float]], dict]:
    """Return top-k label probabilities and extraction metadata for one video."""
    if not video.exists():
        raise FileNotFoundError(f"video not found: {video}")

    model_path = artifacts / "sign_classifier.joblib"
    encoder_path = artifacts / "label_encoder.joblib"
    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            f"missing classifier artifacts in {artifacts}. Run train_classifier.py first."
        )

    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)

    hand_model = download_if_missing(
        HAND_MODEL_URL, models_dir / "hand_landmarker.task"
    )
    pose_model = download_if_missing(
        POSE_MODEL_URL, models_dir / "pose_landmarker_lite.task"
    )

    BaseOptions = mp_python.BaseOptions
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    with vision.PoseLandmarker.create_from_options(
        pose_options
    ) as pose_landmarker, vision.HandLandmarker.create_from_options(
        hand_options
    ) as hand_landmarker:
        feats, meta, _ = extract_video(
            video,
            pose_landmarker,
            hand_landmarker,
            frame_stride,
            timestamp_offset_ms=0,
        )

    if feats.shape[0] == 0 or feats.shape[1] != FEAT_DIM:
        raise RuntimeError(f"bad features shape {feats.shape}")

    x = sequence_to_features(feats).reshape(1, -1)
    proba = clf.predict_proba(x)[0]
    k = min(top_k, len(proba))
    idx = np.argsort(proba)[::-1][:k]
    top = [(str(le.classes_[i]), float(proba[i])) for i in idx]
    return top, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict sign gloss from a video")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args()

    try:
        print(f"Extracting landmarks from {args.video.name} ...")
        top_preds, meta = predict_video_topk(
            video=args.video,
            artifacts=args.artifacts,
            models_dir=args.models_dir,
            top_k=args.top_k,
            frame_stride=args.frame_stride,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"frames_kept={meta['frames_kept']}")
    print("--- top predictions ---")
    for rank, (label, prob) in enumerate(top_preds, start=1):
        print(f"{rank}. {label:20s}  {prob*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
