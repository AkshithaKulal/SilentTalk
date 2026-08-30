"""Run predict_sign.py across every video in verification_set/ and report accuracy."""
import sys
import re
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VERIFICATION_DIR = Path("isl_recognition/verification_set")
ARTIFACTS = Path("isl_recognition/transfer_pack")
PREDICT_SCRIPT = Path("isl_recognition/predict_sign.py")
PYTHON = sys.executable
TOP_K = 5


def extract_true_label(folder_name: str) -> str:
    """Strip number prefix: '48. Hello' → 'Hello'"""
    return re.sub(r"^\d+\.\s*", "", folder_name).strip()


def run_prediction(video_path: Path) -> tuple[str, float, list[str]] | None:
    """Run predict_sign.py on a video, return (top1_label, top1_conf, top5_labels)."""
    result = subprocess.run(
        [PYTHON, str(PREDICT_SCRIPT), "--video", str(video_path),
         "--artifacts", str(ARTIFACTS), "--top-k", str(TOP_K)],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    output = result.stdout + result.stderr
    predictions = []
    for line in output.splitlines():
        m = re.match(r"\s*\d+\.\s+(.+?)\s{2,}(\d+\.\d+)%", line)
        if m:
            predictions.append((m.group(1).strip(), float(m.group(2)) / 100))
    if not predictions:
        return None
    top1_label, top1_conf = predictions[0]
    top_labels = [p[0] for p in predictions]
    return top1_label, top1_conf, top_labels


# ── MAIN ──────────────────────────────────────────────────────────────────────
rows = []
errors = []

sign_folders = sorted(VERIFICATION_DIR.iterdir())
total_videos = sum(
    len([v for v in f.iterdir() if v.suffix.lower() in (".mp4", ".mov", ".avi")])
    for f in sign_folders if f.is_dir()
)
print(f"Running predictions on {total_videos} videos across {len(sign_folders)} signs...\n")

for folder in sign_folders:
    if not folder.is_dir():
        continue
    true_label = extract_true_label(folder.name)
    videos = sorted([v for v in folder.iterdir() if v.suffix.lower() in (".mp4", ".mov", ".avi")])

    for video in videos:
        res = run_prediction(video)
        if res is None:
            errors.append(f"FAILED: {video}")
            continue
        top1_label, top1_conf, top5_labels = res
        top1_correct = top1_label.lower() == true_label.lower()
        top3_correct = true_label.lower() in [l.lower() for l in top5_labels[:3]]
        rows.append({
            "true": true_label,
            "video": video.name,
            "pred": top1_label,
            "conf": top1_conf,
            "top1_ok": top1_correct,
            "top3_ok": top3_correct,
        })
        status = "✓" if top1_correct else ("~" if top3_correct else "✗")
        print(f"  [{status}] {true_label:<20} → {top1_label:<20} {top1_conf*100:5.1f}%  ({video.name})")

# ── FULL TABLE ────────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"{'True Label':<20} {'Video':<25} {'Predicted':<22} {'Conf':>6}  Top-1  Top-3")
print("=" * 100)
for r in rows:
    t1 = "YES" if r["top1_ok"] else "NO "
    t3 = "YES" if r["top3_ok"] else "NO "
    print(f"{r['true']:<20} {r['video']:<25} {r['pred']:<22} {r['conf']*100:5.1f}%   {t1}    {t3}")

# ── PER-WORD ACCURACY ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PER-SIGN ACCURACY")
print("=" * 70)
print(f"{'Sign':<22} {'Videos':>6}  {'Top-1':>6}  {'Top-3':>6}  {'Top-1%':>7}  {'Top-3%':>7}")
print("-" * 70)

from collections import defaultdict
per_sign: dict[str, list] = defaultdict(list)
for r in rows:
    per_sign[r["true"]].append(r)

for sign in sorted(per_sign):
    sign_rows = per_sign[sign]
    n = len(sign_rows)
    t1 = sum(1 for r in sign_rows if r["top1_ok"])
    t3 = sum(1 for r in sign_rows if r["top3_ok"])
    print(f"{sign:<22} {n:>6}  {t1:>6}  {t3:>6}  {t1/n*100:>6.0f}%  {t3/n*100:>6.0f}%")

# ── OVERALL ACCURACY ──────────────────────────────────────────────────────────
n = len(rows)
top1_total = sum(1 for r in rows if r["top1_ok"])
top3_total = sum(1 for r in rows if r["top3_ok"])
print("\n" + "=" * 70)
print(f"OVERALL  |  Videos: {n}  |  Top-1: {top1_total}/{n} ({top1_total/n*100:.1f}%)  |  Top-3: {top3_total}/{n} ({top3_total/n*100:.1f}%)")
print("=" * 70)

if errors:
    print(f"\nFAILED ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
