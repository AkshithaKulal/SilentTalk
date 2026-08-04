#!/usr/bin/env python3
"""Summarize extracted INCLUDE landmark .npy files (counts per label)."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def clean_label(raw: str) -> str:
    # INCLUDE folders look like "66. Sunday" / "47. they"
    m = re.match(r"^\s*\d+\.\s*(.+)\s*$", raw)
    return (m.group(1) if m else raw).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--landmarks",
        type=Path,
        default=Path(__file__).resolve().parent / "landmarks",
    )
    args = parser.parse_args()

    npy_files = sorted(args.landmarks.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files in {args.landmarks}")
        return 1

    labels: list[str] = []
    frames: list[int] = []
    for npy in npy_files:
        meta_path = npy.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw = meta.get("label") or "UNKNOWN"
        else:
            raw = "UNKNOWN"
        labels.append(clean_label(raw))
        arr = np.load(npy)
        frames.append(int(arr.shape[0]))

    counts = Counter(labels)
    print(f"files={len(npy_files)} labels={len(counts)}")
    print(f"frames_kept: min={min(frames)} median={int(np.median(frames))} max={max(frames)}")
    print("--- per label ---")
    for label, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{n:4d}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
