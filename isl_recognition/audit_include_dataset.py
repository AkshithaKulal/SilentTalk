#!/usr/bin/env python3
"""Audit INCLUDE extracted video dataset and emit a planning-ready report.

Usage:
  python audit_include_dataset.py --root F:\\include_dataset\\extracted
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}


def clean_label(raw: str) -> str:
    match = re.match(r"^\s*\d+\.\s*(.+)\s*$", raw)
    return (match.group(1) if match else raw).strip()


def try_open_video(path: Path) -> tuple[bool, str]:
    """Try opening one frame with OpenCV; return (ok, error_message)."""
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        return False, f"opencv-unavailable: {exc}"

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False, "cannot-open"
    ok, _frame = cap.read()
    cap.release()
    if not ok:
        return False, "cannot-read-frame"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit INCLUDE extracted dataset")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"F:\include_dataset\extracted"),
        help="Root folder of extracted INCLUDE videos",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "include_audit_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=5,
        help="Flag labels with fewer than this many clips",
    )
    parser.add_argument(
        "--check-open-limit",
        type=int,
        default=200,
        help="Try opening up to N video files to detect corruption (0 disables)",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"ERROR: dataset root not found: {args.root}")
        return 1

    all_files = sorted(p for p in args.root.rglob("*") if p.is_file())
    video_files = [p for p in all_files if p.suffix.lower() in VIDEO_EXTS]
    non_video_files = [p for p in all_files if p.suffix.lower() not in VIDEO_EXTS]

    if not video_files:
        print(f"ERROR: no video files found under {args.root}")
        return 1

    category_part_to_labels: dict[str, Counter] = defaultdict(Counter)
    label_counts: Counter = Counter()
    depth_issues: list[str] = []

    for path in video_files:
        rel = path.relative_to(args.root)
        parts = rel.parts
        if len(parts) < 2:
            depth_issues.append(str(rel))
        category_part = parts[0] if parts else "UNKNOWN"
        word_folder = parts[-2] if len(parts) >= 2 else path.parent.name
        label = clean_label(word_folder)
        category_part_to_labels[category_part][label] += 1
        label_counts[label] += 1

    counts = list(label_counts.values())
    unique_labels = sorted(label_counts)
    low_resource_labels = sorted([lab for lab, n in label_counts.items() if n < args.min_per_class])

    open_check = {
        "enabled": args.check_open_limit > 0,
        "checked": 0,
        "failed": 0,
        "failed_examples": [],
    }

    if args.check_open_limit > 0:
        to_check = video_files[: args.check_open_limit]
        for path in to_check:
            ok, error = try_open_video(path)
            open_check["checked"] += 1
            if not ok:
                open_check["failed"] += 1
                if len(open_check["failed_examples"]) < 20:
                    open_check["failed_examples"].append(
                        {
                            "path": str(path),
                            "error": error,
                        }
                    )

    report = {
        "root": str(args.root),
        "total_files": len(all_files),
        "total_video_files": len(video_files),
        "total_non_video_files": len(non_video_files),
        "category_part_count": len(category_part_to_labels),
        "unique_label_count": len(unique_labels),
        "unique_labels": unique_labels,
        "clip_stats": {
            "min_per_label": int(min(counts)),
            "median_per_label": int(sorted(counts)[len(counts) // 2]),
            "max_per_label": int(max(counts)),
        },
        "labels_below_min_per_class": low_resource_labels,
        "labels_below_min_per_class_count": len(low_resource_labels),
        "label_counts": dict(sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))),
        "category_parts": {
            cat: dict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))
            for cat, counter in sorted(category_part_to_labels.items())
        },
        "path_depth_issues": depth_issues[:100],
        "path_depth_issue_count": len(depth_issues),
        "open_check": open_check,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Root: {args.root}")
    print(f"Videos: {len(video_files)}")
    print(f"Unique labels: {len(unique_labels)}")
    print(
        "Clips/label min/median/max: "
        f"{report['clip_stats']['min_per_label']}/"
        f"{report['clip_stats']['median_per_label']}/"
        f"{report['clip_stats']['max_per_label']}"
    )
    print(f"Labels below {args.min_per_class}: {len(low_resource_labels)}")
    if open_check["enabled"]:
        print(
            f"Open-check failures: {open_check['failed']}/{open_check['checked']} "
            "(sampled)"
        )
    print(f"Saved report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
