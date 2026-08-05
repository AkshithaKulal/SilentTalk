#!/usr/bin/env python3
"""Full inventory of INCLUDE extracted video folders.

Example:
  python inventory_include.py --root F:\\include_dataset\\extracted
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}


def clean_label(name: str) -> str:
    m = re.match(r"^\s*\d+\.\s*(.+)\s*$", name)
    return (m.group(1) if m else name).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"F:\include_dataset\extracted"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "include_inventory.json",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"ERROR: root not found: {args.root}")
        return 1

    # category_part (e.g. Greetings_1of2) -> word -> list of videos
    tree: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    total_videos = 0

    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = path.relative_to(args.root)
        parts = rel.parts
        # Expected: CategoryPart / Category / N. Word / file.MOV
        category_part = parts[0] if len(parts) >= 1 else "UNKNOWN"
        word_folder = parts[-2] if len(parts) >= 2 else path.parent.name
        word = clean_label(word_folder)
        tree[category_part][word].append(str(path))
        total_videos += 1

    words_all = sorted({w for cat in tree.values() for w in cat})
    print(f"Root: {args.root}")
    print(f"Category parts: {len(tree)}")
    print(f"Unique words/signs: {len(words_all)}")
    print(f"Total videos: {total_videos}")
    print()

    for cat in sorted(tree):
        words = tree[cat]
        n_clips = sum(len(v) for v in words.values())
        print(f"=== {cat}  ({len(words)} words, {n_clips} videos) ===")
        for word in sorted(words, key=str.lower):
            print(f"  {len(words[word]):3d}  {word}")
        print()

    print("=== ALL UNIQUE WORDS (alphabetical) ===")
    for i, w in enumerate(words_all, start=1):
        clips = sum(len(tree[c][w]) for c in tree if w in tree[c])
        print(f"{i:3d}. {w}  ({clips} clips)")

    payload = {
        "root": str(args.root),
        "category_parts": {
            cat: {w: len(vids) for w, vids in sorted(words.items())}
            for cat, words in sorted(tree.items())
        },
        "unique_words": words_all,
        "unique_word_count": len(words_all),
        "total_videos": total_videos,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
