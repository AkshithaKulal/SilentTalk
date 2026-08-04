#!/usr/bin/env python3
"""Extract MediaPipe hand + pose landmarks from INCLUDE videos.

Designed for mediapipe>=0.10.30 / 1.0 Tasks API (not legacy mp.solutions).

Example (smoke test first):
  python extract_landmarks.py --input F:\\include_dataset\\extracted --limit 3

Full run:
  python extract_landmarks.py --input F:\\include_dataset\\extracted
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

NUM_POSE = 33
NUM_HAND = 21
NUM_HANDS = 2
FEAT_DIM = NUM_POSE * 3 + NUM_HANDS * NUM_HAND * 3  # 225


def download_if_missing(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest}")
    return dest


def find_videos(root: Path) -> list[Path]:
    videos = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    return sorted(videos)


def label_from_path(video: Path, input_root: Path) -> str:
    """INCLUDE layout is usually .../<WordName>/<clip>.mp4 — use parent folder."""
    try:
        rel = video.relative_to(input_root)
        if len(rel.parts) >= 2:
            return rel.parts[-2]
    except ValueError:
        pass
    return video.parent.name


def landmarks_to_xyz(landmarks, expected: int) -> np.ndarray:
    out = np.zeros((expected, 3), dtype=np.float32)
    if not landmarks:
        return out
    n = min(expected, len(landmarks))
    for i in range(n):
        lm = landmarks[i]
        out[i, 0] = lm.x
        out[i, 1] = lm.y
        out[i, 2] = lm.z
    return out


def frame_feature(pose_result, hand_result) -> np.ndarray:
    pose = np.zeros((NUM_POSE, 3), dtype=np.float32)
    if pose_result.pose_landmarks:
        pose = landmarks_to_xyz(pose_result.pose_landmarks[0], NUM_POSE)

    left = np.zeros((NUM_HAND, 3), dtype=np.float32)
    right = np.zeros((NUM_HAND, 3), dtype=np.float32)

    if hand_result.hand_landmarks:
        for hand_lms, handedness in zip(
            hand_result.hand_landmarks, hand_result.handedness
        ):
            name = ""
            if handedness:
                name = (handedness[0].category_name or "").lower()
            arr = landmarks_to_xyz(hand_lms, NUM_HAND)
            if name.startswith("left"):
                left = arr
            elif name.startswith("right"):
                right = arr
            else:
                # Unknown handedness: fill first empty slot
                if not left.any():
                    left = arr
                else:
                    right = arr

    return np.concatenate([pose.reshape(-1), left.reshape(-1), right.reshape(-1)])


def extract_video(
    video_path: Path,
    pose_landmarker,
    hand_landmarker,
    frame_stride: int,
    timestamp_offset_ms: int = 0,
) -> tuple[np.ndarray, dict, int]:
    """Return features, meta, and next timestamp offset (ms) for VIDEO mode.

    MediaPipe VIDEO mode requires timestamps to keep rising across the whole
    landmarker lifetime — including across separate video files — so we carry
    an offset from the previous clip.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    feats: list[np.ndarray] = []
    frame_idx = 0
    kept = 0
    last_ts = timestamp_offset_ms

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # MediaPipe Image wants contiguous uint8 RGB
        frame_rgb = np.ascontiguousarray(frame_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Timestamp must be monotonically increasing in VIDEO mode
        if fps > 1e-3:
            local_ms = int(round(frame_idx * (1000.0 / fps)))
        else:
            local_ms = frame_idx * 33
        timestamp_ms = timestamp_offset_ms + local_ms
        if timestamp_ms <= last_ts:
            timestamp_ms = last_ts + 1
        last_ts = timestamp_ms

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        feats.append(frame_feature(pose_result, hand_result))
        kept += 1
        frame_idx += 1

    cap.release()
    arr = np.stack(feats, axis=0) if feats else np.zeros((0, FEAT_DIM), dtype=np.float32)
    meta = {
        "source": str(video_path),
        "fps": fps,
        "frame_count_total": total,
        "frames_kept": kept,
        "frame_stride": frame_stride,
        "feature_dim": FEAT_DIM,
        "layout": "pose33_xyz + left_hand21_xyz + right_hand21_xyz",
    }
    # Leave a gap so the next video clearly continues the timeline
    return arr, meta, last_ts + 1000


def out_stem_for(video: Path, input_root: Path) -> str:
    try:
        rel = video.relative_to(input_root)
        return "__".join(rel.with_suffix("").parts)
    except ValueError:
        return video.stem


def main() -> int:
    parser = argparse.ArgumentParser(description="INCLUDE → MediaPipe landmarks")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"F:\include_dataset\extracted"),
        help="Root folder of extracted INCLUDE videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "landmarks",
        help="Where to write .npy + .json",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="Directory for .task model files",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only N videos (0=all)")
    parser.add_argument("--frame-stride", type=int, default=2, help="Keep every Nth frame")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have a .npy output",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    hand_model = download_if_missing(HAND_MODEL_URL, args.models_dir / "hand_landmarker.task")
    pose_model = download_if_missing(
        POSE_MODEL_URL, args.models_dir / "pose_landmarker_lite.task"
    )

    videos = find_videos(args.input)
    if not videos:
        print(f"ERROR: no videos under {args.input}", file=sys.stderr)
        return 1

    if args.limit > 0:
        videos = videos[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Videos to process: {len(videos)}")
    print(f"Output dir: {args.output}")

    BaseOptions = mp_python.BaseOptions
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    ok_count = 0
    fail_count = 0
    timestamp_offset_ms = 0

    with vision.PoseLandmarker.create_from_options(
        pose_options
    ) as pose_landmarker, vision.HandLandmarker.create_from_options(
        hand_options
    ) as hand_landmarker:
        for i, video in enumerate(videos, start=1):
            stem = out_stem_for(video, args.input)
            npy_path = args.output / f"{stem}.npy"
            json_path = args.output / f"{stem}.json"

            if args.skip_existing and npy_path.exists():
                print(f"[{i}/{len(videos)}] SKIP existing {npy_path.name}")
                ok_count += 1
                continue

            label = label_from_path(video, args.input)
            print(f"[{i}/{len(videos)}] {label} <- {video.name}")
            try:
                feats, meta, timestamp_offset_ms = extract_video(
                    video,
                    pose_landmarker,
                    hand_landmarker,
                    args.frame_stride,
                    timestamp_offset_ms=timestamp_offset_ms,
                )
                meta["label"] = label
                meta["output_npy"] = str(npy_path)
                np.save(npy_path, feats)
                json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(
                    f"    saved {npy_path.name} shape={feats.shape} "
                    f"(frames_kept={meta['frames_kept']})"
                )
                ok_count += 1
            except Exception as exc:  # noqa: BLE001 — keep batch running
                fail_count += 1
                print(f"    FAIL: {exc}", file=sys.stderr)

    print(f"Done. ok={ok_count} fail={fail_count} out={args.output}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
