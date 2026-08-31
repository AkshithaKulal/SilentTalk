#!/usr/bin/env python3
"""Standalone INCLUDE folder audit.

Copy THIS ONE FILE into the office INCLUDE folder (the folder that contains
the unzipped videos, usually F:\\Include_dataset\\extracted) and run:

    python audit_include_folder.py

No SilentTalk repo, no pip packages required (Python 3.8+ stdlib only).

It prints a summary and writes include_folder_audit.json next to the script.
Send that JSON back so we can write the training scripts against the real layout.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}

EXPECTED_CATEGORIES = [
    "Adjectives",
    "Animals",
    "Clothes",
    "Colours",
    "Days_and_Time",
    "Electronics",
    "Greetings",
    "Home",
    "Jobs",
    "Means_of_Transportation",
    "People",
    "Places",
    "Pronouns",
    "Seasons",
    "Society",
]

# Official INCLUDE-50 display names (paper sanity subset).
INCLUDE50 = {
    "Bank",
    "big large",
    "Bird",
    "Black",
    "Boy",
    "Brother",
    "Car",
    "Cell phone",
    "Court",
    "Cow",
    "Death",
    "Dog",
    "dry",
    "Election",
    "Fall",
    "Fan",
    "Father",
    "Girl",
    "good",
    "Good Morning",
    "happy",
    "Hat",
    "Hello",
    "hot",
    "House",
    "I",
    "it",
    "long",
    "loud",
    "Monday",
    "new",
    "Paint",
    "Pen",
    "Priest",
    "quiet",
    "Red",
    "Shoes",
    "short",
    "small little",
    "Store or Shop",
    "Summer",
    "Teacher",
    "Thank you",
    "Time",
    "train ticket",
    "T-Shirt",
    "White",
    "Window",
    "Year",
    "you (plural)",
}

ZIP_PART = re.compile(r"^(.+)_(\d+)of(\d+)$", re.I)
NUM_WORD = re.compile(r"^\s*\d+\.\s*(.+)\s*$")


def clean_label(raw: str) -> str:
    raw = (raw or "").strip()
    m = NUM_WORD.match(raw)
    return (m.group(1) if m else raw).strip()


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_root(start: Path) -> Path:
    """Prefer a folder that actually contains INCLUDE videos."""
    candidates = [start]
    for name in ("extracted", "Extracted", "INCLUDE", "include", "videos"):
        candidates.append(start / name)
    if start.name.lower() != "extracted" and (start.parent / "extracted").exists():
        candidates.append(start.parent / "extracted")

    scored: list[tuple[int, Path]] = []
    for cand in candidates:
        if not cand.is_dir():
            continue
        n = 0
        for p in cand.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                n += 1
                if n >= 20:
                    break
        if n:
            scored.append((n, cand))
    if scored:
        scored.sort(key=lambda x: (-x[0], len(x[1].parts)))
        return scored[0][1]
    return start


def word_folder_and_extra(video: Path, root: Path) -> tuple[str, bool, str]:
    """Return (raw word folder, is_extra_subfolder, relative posix path)."""
    rel = video.relative_to(root)
    parts = list(rel.parts)
    extra = False
    if len(parts) >= 2 and parts[-2].strip().lower() == "extra":
        extra = True
        word_raw = parts[-3] if len(parts) >= 3 else parts[-2]
    elif len(parts) >= 2:
        word_raw = parts[-2]
    else:
        word_raw = video.parent.name
    return word_raw, extra, rel.as_posix()


def category_from_parts(parts: tuple[str, ...]) -> str:
    names = {c.lower(): c for c in EXPECTED_CATEGORIES}
    for part in parts:
        key = part.lower()
        if key in names:
            return names[key]
        m = ZIP_PART.match(part)
        if m:
            base = m.group(1)
            if base.lower() in names:
                return names[base.lower()]
    return parts[0] if parts else "UNKNOWN"


def collect_videos(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def main() -> int:
    here = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
    start = Path.cwd() if Path.cwd() != here else here
    # If the file was copied into the dataset folder, prefer that folder.
    start = Path(__file__).resolve().parent

    root = find_root(start)
    print("=" * 72)
    print("INCLUDE FOLDER AUDIT")
    print("=" * 72)
    print(f"Script dir : {start}")
    print(f"Scan root  : {root}")
    if not root.exists():
        print("ERROR: folder does not exist.")
        return 1

    videos = collect_videos(root)
    print(f"Video files: {len(videos)}   (official INCLUDE is ~4292)")
    if not videos:
        print()
        print("No videos found. Put this script inside the extracted INCLUDE")
        print("folder (the one that contains Greetings_1of2, People_1of5, ...)")
        print("or run:  python audit_include_folder.py")
        print("from that folder.")
        return 1

    label_counts: Counter[str] = Counter()
    cat_counts: Counter[str] = Counter()
    cat_labels: dict[str, Counter[str]] = defaultdict(Counter)
    extra_examples: list[str] = []
    extra_count = 0
    zip_prefix_count = 0
    depth_hist: Counter[int] = Counter()
    samples: list[str] = []
    exts: Counter[str] = Counter()

    for video in videos:
        rel = video.relative_to(root)
        depth_hist[len(rel.parts)] += 1
        exts[video.suffix.lower()] += 1
        word_raw, is_extra, rel_posix = word_folder_and_extra(video, root)
        label = clean_label(word_raw)
        cat = category_from_parts(rel.parts)
        label_counts[label] += 1
        cat_counts[cat] += 1
        cat_labels[cat][label] += 1
        if is_extra:
            extra_count += 1
            if len(extra_examples) < 8:
                extra_examples.append(rel_posix)
        if ZIP_PART.match(rel.parts[0] if rel.parts else ""):
            zip_prefix_count += 1
        if len(samples) < 8:
            samples.append(rel_posix)

    labels = sorted(label_counts)
    include50_norm = {norm(x): x for x in INCLUDE50}
    have50 = []
    miss50 = []
    for display in sorted(INCLUDE50, key=str.lower):
        if norm(display) in {norm(x) for x in labels}:
            have50.append(display)
        else:
            miss50.append(display)

    found_cats = sorted(cat_counts)
    missing_cats = [c for c in EXPECTED_CATEGORIES if c.lower() not in {x.lower() for x in found_cats}]
    extra_cats = [c for c in found_cats if c.lower() not in {x.lower() for x in EXPECTED_CATEGORIES}]
    low = sorted([(lab, n) for lab, n in label_counts.items() if n < 5], key=lambda x: (x[1], x[0]))

    print()
    print("--- vs official INCLUDE ---")
    print(f"  unique words     : {len(labels):4d}   (expected 263)")
    print(f"  categories       : {len(found_cats):4d}   (expected 15)")
    print(f"  zip-part prefixes: {zip_prefix_count}/{len(videos)} videos start with Category_NofM")
    print(f"  Extra/ subfolders: {extra_count} clips")
    print(f"  INCLUDE-50 found : {len(have50)}/50")
    print(f"  extensions       : {dict(exts)}")

    print()
    print("--- sample paths (first 8) ---")
    for s in samples:
        print(f"  {s}")

    print()
    print("--- folder depth (path parts under root) ---")
    for d, n in sorted(depth_hist.items()):
        print(f"  depth {d}: {n} files")

    print()
    print("--- category video counts ---")
    for cat in EXPECTED_CATEGORIES + [c for c in extra_cats if c not in EXPECTED_CATEGORIES]:
        n = cat_counts.get(cat, 0)
        nlab = len(cat_labels.get(cat, {}))
        mark = "OK" if n else "MISSING"
        print(f"  {cat:<28}  videos={n:4d}  words={nlab:3d}  {mark}")

    if missing_cats:
        print()
        print("MISSING CATEGORIES:", ", ".join(missing_cats))
    if extra_cats:
        print("UNEXPECTED TOP FOLDERS:", ", ".join(extra_cats))

    if extra_examples:
        print()
        print("--- Extra/ examples (label must be the WORD folder, not Extra) ---")
        for s in extra_examples:
            print(f"  {s}")

    print()
    print("--- words with fewer than 5 clips ---")
    if not low:
        print("  none")
    else:
        for lab, n in low[:40]:
            print(f"  {n:3d}  {lab}")
        if len(low) > 40:
            print(f"  ... {len(low) - 40} more")

    print()
    print("--- INCLUDE-50 missing from this folder ---")
    if not miss50:
        print("  none (all 50 word names found)")
    else:
        print("  " + ", ".join(miss50))

    print()
    print("--- all unique words ---")
    for i, lab in enumerate(labels, start=1):
        print(f"  {i:3d}. {lab:<28} {label_counts[lab]:3d} clips")

    report = {
        "scan_root": str(root),
        "video_count": len(videos),
        "unique_word_count": len(labels),
        "expected_videos": 4292,
        "expected_words": 263,
        "zip_part_prefix_videos": zip_prefix_count,
        "extra_subfolder_clips": extra_count,
        "extra_examples": extra_examples,
        "sample_paths": samples,
        "depth_histogram": {str(k): v for k, v in sorted(depth_hist.items())},
        "extensions": dict(exts),
        "categories_found": found_cats,
        "categories_missing": missing_cats,
        "category_video_counts": dict(sorted(cat_counts.items(), key=lambda x: (-x[1], x[0]))),
        "category_word_counts": {c: dict(sorted(counter.items())) for c, counter in cat_labels.items()},
        "include50_found": have50,
        "include50_missing": miss50,
        "labels_below_5": [{"label": lab, "clips": n} for lab, n in low],
        "label_counts": dict(sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))),
        "unique_labels": labels,
    }

    out = start / "include_folder_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print("=" * 72)
    print(f"Saved {out}")
    print("Copy that JSON (or this terminal output) back to the laptop.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
