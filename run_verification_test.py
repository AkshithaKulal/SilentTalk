#!/usr/bin/env python3
"""Run prediction on all verification_set clips and print accuracy report."""
import sys, time, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "isl_recognition"))

import joblib, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from extract_landmarks import download_if_missing, extract_video, HAND_MODEL_URL, POSE_MODEL_URL
from train_classifier import sequence_to_features

MODELS    = Path("isl_recognition/models")
ARTIFACTS = Path("isl_recognition/transfer_pack")
VSET      = Path("isl_recognition/verification_set")

print("Loading classifier...", flush=True)
clf = joblib.load(ARTIFACTS / "sign_classifier.joblib")
le  = joblib.load(ARTIFACTS / "label_encoder.joblib")
print(f"Classifier ready — {len(le.classes_)} classes\n", flush=True)

BaseOptions = mp_python.BaseOptions
pose_opts = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_lite.task")),
    running_mode=vision.RunningMode.VIDEO, num_poses=1)
hand_opts = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
    running_mode=vision.RunningMode.VIDEO, num_hands=2)

correct = 0
top3_correct = 0
total   = 0
results = []

with vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm, \
     vision.HandLandmarker.create_from_options(hand_opts) as hand_lm:

    ts_offset = 0
    for folder in sorted(VSET.iterdir()):
        if not folder.is_dir():
            continue
        label = re.sub(r"^\d+\.\s*", "", folder.name).strip()
        vids = sorted([v for v in folder.iterdir()
                       if v.suffix.lower() in (".mp4", ".mov", ".avi")])
        if not vids:
            continue
        video = vids[0]
        print(f"  Testing: {label:<20} ({video.name})", end="  ", flush=True)

        t0 = time.perf_counter()
        try:
            feats, meta, ts_offset = extract_video(
                video, pose_lm, hand_lm, frame_stride=2, timestamp_offset_ms=ts_offset)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"label": label, "top1": "ERROR", "conf": 0,
                            "top5": [], "time_ms": 0, "ok": False, "ok3": False})
            total += 1
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if feats.shape[0] == 0:
            print("NO FEATURES")
            results.append({"label": label, "top1": "NO FEATURES", "conf": 0,
                            "top5": [], "time_ms": int(elapsed_ms), "ok": False, "ok3": False})
            total += 1
            continue

        x = sequence_to_features(feats).reshape(1, -1)
        proba = clf.predict_proba(x)[0]
        idx = np.argsort(proba)[::-1][:5]
        top5 = [(str(le.classes_[i]), round(float(proba[i]) * 100, 1)) for i in idx]

        top1_label = top5[0][0]
        top1_conf  = top5[0][1]
        top3_labels = [t[0].lower() for t in top5[:3]]

        ok  = top1_label.lower() == label.lower()
        ok3 = label.lower() in top3_labels
        correct      += int(ok)
        top3_correct += int(ok3)
        total        += 1

        mark = "CORRECT" if ok else (f"TOP-3 ({top5[1][0]})" if ok3 else f"WRONG -> {top1_label}")
        print(f"{top1_conf:5.1f}%  {mark}  [{int(elapsed_ms)}ms]")

        results.append({
            "label": label, "top1": top1_label, "conf": top1_conf,
            "top5": top5, "time_ms": int(elapsed_ms), "ok": ok, "ok3": ok3
        })

print()
print("=" * 72)
print(f"  VERIFICATION SET RESULTS  —  {total} clips")
print("=" * 72)
print(f"  {'SIGN':<22} {'PREDICTED':<22} {'CONF':>6}  {'TOP1':>6}  {'IN TOP3':>7}  {'ms':>6}")
print("-" * 72)
for r in results:
    t1 = "  OK " if r["ok"]  else "  -- "
    t3 = "  OK " if r["ok3"] else "  -- "
    print(f"  {r['label']:<22} {r['top1']:<22} {r['conf']:>5.1f}%  {t1}  {t3}  {r['time_ms']:>5}ms")
print("=" * 72)
print(f"  Top-1 accuracy : {correct}/{total} = {correct/total*100:.1f}%")
print(f"  Top-3 accuracy : {top3_correct}/{total} = {top3_correct/total*100:.1f}%")
print("=" * 72)

if any(not r["ok"] for r in results):
    print("\nMISCLASSIFIED:")
    for r in results:
        if not r["ok"]:
            t3 = " | ".join([f"{t[0]} {t[1]}%" for t in r["top5"][:3]])
            print(f"  {r['label']:<20} -> {r['top1']} ({r['conf']}%)   top3: {t3}")
