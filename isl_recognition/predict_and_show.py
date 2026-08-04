#!/usr/bin/env python3
"""Predict ISL gloss and play the video with landmarks + label overlay.

Example:
  python predict_and_show.py --video path\\to\\clip.MOV
Keys: q / Esc = quit, SPACE = pause
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from extract_landmarks import (
    FEAT_DIM,
    HAND_MODEL_URL,
    POSE_MODEL_URL,
    download_if_missing,
    frame_feature,
)
from train_classifier import sequence_to_features

try:
    from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
except ImportError:  # older / alternate layouts
    drawing_utils = None
    drawing_styles = None


def draw_overlay(frame_bgr, pose_result, hand_result, title: str, lines: list[str]):
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    def _dots(landmarks, color):
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(annotated, (x, y), 3, color, -1)

    drew = False
    if drawing_utils is not None:
        try:
            if pose_result.pose_landmarks:
                pose_conn = getattr(
                    vision.PoseLandmarksConnections, "POSE_LANDMARKS", None
                ) or getattr(vision.PoseLandmarksConnections, "POSE_CONNECTIONS", None)
                drawing_utils.draw_landmarks(
                    annotated,
                    pose_result.pose_landmarks[0],
                    pose_conn,
                    drawing_styles.get_default_pose_landmarks_style()
                    if drawing_styles
                    else None,
                )
            if hand_result.hand_landmarks:
                hand_conn = getattr(
                    vision.HandLandmarksConnections, "HAND_CONNECTIONS", None
                )
                for hand_lms in hand_result.hand_landmarks:
                    drawing_utils.draw_landmarks(
                        annotated,
                        hand_lms,
                        hand_conn,
                        drawing_styles.get_default_hand_landmarks_style()
                        if drawing_styles
                        else None,
                        drawing_styles.get_default_hand_connections_style()
                        if drawing_styles
                        else None,
                    )
            drew = True
        except Exception:
            drew = False

    if not drew:
        if pose_result.pose_landmarks:
            _dots(pose_result.pose_landmarks[0], (0, 255, 0))
        if hand_result.hand_landmarks:
            for hand_lms in hand_result.hand_landmarks:
                _dots(hand_lms, (0, 128, 255))

    # Banner
    cv2.rectangle(annotated, (0, 0), (w, 90 + 22 * len(lines)), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for i, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (12, 60 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 255, 80),
            2,
            cv2.LINE_AA,
        )
    return annotated


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save annotated .mp4",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1

    model_path = args.artifacts / "sign_classifier.joblib"
    encoder_path = args.artifacts / "label_encoder.joblib"
    if not model_path.exists() or not encoder_path.exists():
        print(f"ERROR: train classifier first; missing {args.artifacts}", file=sys.stderr)
        return 1

    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)

    hand_model = download_if_missing(
        HAND_MODEL_URL, args.models_dir / "hand_landmarker.task"
    )
    pose_model = download_if_missing(
        POSE_MODEL_URL, args.models_dir / "pose_landmarker_lite.task"
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

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        return 1

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    delay = max(1, int(1000 / fps))

    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, fps, (width, height))

    feats: list[np.ndarray] = []
    frames_out: list[np.ndarray] = []
    frame_idx = 0
    last_ts = -1

    print("Pass 1: extract landmarks ...")
    with vision.PoseLandmarker.create_from_options(
        pose_options
    ) as pose_landmarker, vision.HandLandmarker.create_from_options(
        hand_options
    ) as hand_landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            use = frame_idx % args.frame_stride == 0
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(round(frame_idx * (1000.0 / fps))) if fps > 1e-3 else frame_idx * 33
            if timestamp_ms <= last_ts:
                timestamp_ms = last_ts + 1
            last_ts = timestamp_ms

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            if use:
                feats.append(frame_feature(pose_result, hand_result))

            # Temporary overlay without prediction text yet
            frames_out.append((frame_bgr, pose_result, hand_result))
            frame_idx += 1

    cap.release()

    if not feats:
        print("ERROR: no frames extracted", file=sys.stderr)
        return 1

    seq = np.stack(feats, axis=0)
    if seq.shape[1] != FEAT_DIM:
        print(f"ERROR: bad feature dim {seq.shape}", file=sys.stderr)
        return 1

    x = sequence_to_features(seq).reshape(1, -1)
    proba = clf.predict_proba(x)[0]
    top_k = min(args.top_k, len(proba))
    idx = np.argsort(proba)[::-1][:top_k]
    lines = [f"{le.classes_[i]}  {proba[i]*100:.1f}%" for i in idx]
    title = f"Pred: {le.classes_[idx[0]]}"

    print("--- top predictions ---")
    for i, line in enumerate(lines, start=1):
        print(f"{i}. {line}")
    print("Playing video window (q / Esc quit, SPACE pause) ...")

    paused = False
    win = "SilentTalk — sign prediction"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for frame_bgr, pose_result, hand_result in frames_out:
        annotated = draw_overlay(frame_bgr, pose_result, hand_result, title, lines)
        if writer is not None:
            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height))
            writer.write(annotated)
        cv2.imshow(win, annotated)
        while True:
            key = cv2.waitKey(0 if paused else delay) & 0xFF
            if key in (ord("q"), 27):
                if writer is not None:
                    writer.release()
                cv2.destroyAllWindows()
                return 0
            if key == ord(" "):
                paused = not paused
                continue
            break

    if writer is not None:
        writer.release()
        print(f"Saved annotated video: {args.save}")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
