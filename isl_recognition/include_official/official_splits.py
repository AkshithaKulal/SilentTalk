"""Download official INCLUDE train/test path lists and match landmark files.

Split files (AI4Bharat):
  Category/N. Word/MVI_xxxx.MOV
Landmark stems look like:
  Seasons_1of1__Seasons__63. Winter__MVI_4997
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

NUM_RE = re.compile(r"^\s*\d+\.\s*(.+)\s*$")
CLIP_RE = re.compile(r"(MVI_\d+)", re.I)

RAW = {
    "train": "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_train.txt",
    "test": "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_test.txt",
}

USER_AGENT = "SilentTalk-INCLUDE-splits/1.0"


def clean_label(raw: str) -> str:
    raw = (raw or "").strip()
    m = NUM_RE.match(raw)
    return (m.group(1) if m else raw).strip()


def parse_split_line(line: str) -> tuple[str, str] | None:
    line = line.strip().replace("\\", "/")
    if not line or line.startswith("#"):
        return None
    parts = [p for p in line.split("/") if p]
    if len(parts) < 2:
        return None
    clip = Path(parts[-1]).stem
    word = clean_label(parts[-2])
    m = CLIP_RE.search(clip)
    if m:
        clip = m.group(1).upper()
    return clip, word.lower()


def download_list(kind: str, dest: Path) -> list[str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100:
        return dest.read_text(encoding="utf-8").splitlines()
    req = urllib.request.Request(RAW[kind], headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    dest.write_text(text, encoding="utf-8")
    return text.splitlines()


def keys_from_file(path: Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_split_line(line)
        if parsed:
            out.add(parsed)
    return out


def item_key(npy: Path, label: str) -> tuple[str, str]:
    m = CLIP_RE.search(npy.stem)
    clip = m.group(1).upper() if m else npy.stem.upper()
    return clip, (label or "").strip().lower()


def load_official_keys(cache_dir: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_lines = download_list("train", cache_dir / "include_train.txt")
    test_lines = download_list("test", cache_dir / "include_test.txt")
    train_keys: set[tuple[str, str]] = set()
    test_keys: set[tuple[str, str]] = set()
    for line in train_lines:
        p = parse_split_line(line)
        if p:
            train_keys.add(p)
    for line in test_lines:
        p = parse_split_line(line)
        if p:
            test_keys.add(p)
    return train_keys, test_keys


def split_items(
    items: list[tuple[Path, str]],
    cache_dir: Path,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], dict]:
    train_keys, test_keys = load_official_keys(cache_dir)
    train, test, unused = [], [], []
    for npy, label in items:
        key = item_key(npy, label)
        if key in test_keys:
            test.append((npy, label))
        elif key in train_keys:
            train.append((npy, label))
        else:
            unused.append((npy, label))
    stats = {
        "official_train_keys": len(train_keys),
        "official_test_keys": len(test_keys),
        "matched_train": len(train),
        "matched_test": len(test),
        "unmatched": len(unused),
        "match_rate": (len(train) + len(test)) / max(len(items), 1),
    }
    # Keep unmatched in train so we do not throw away extra INCLUDE clips.
    train.extend(unused)
    return train, test, stats
