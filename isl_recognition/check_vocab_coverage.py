#!/usr/bin/env python3
"""Compare SilentTalk target vocab against INCLUDE landmark labels.

Run on F: after landmarks exist:
  python check_vocab_coverage.py
  python check_vocab_coverage.py --landmarks .\\landmarks
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Default SilentTalk-style ~50-word domain list (edit if your synopsis list differs)
DEFAULT_50_WORDS = [
    "Hello",
    "Thank you",
    "Please",
    "Sorry",
    "Yes",
    "No",
    "Good",
    "Bye",
    "I",
    "You",
    "He",
    "She",
    "Friend",
    "Doctor",
    "Teacher",
    "Help",
    "Water",
    "Food",
    "Hungry",
    "Thirsty",
    "Pain",
    "Tired",
    "Happy",
    "Sad",
    "Sick",
    "What",
    "Where",
    "When",
    "Who",
    "How",
    "Why",
    "How much",
    "Which",
    "Home",
    "School",
    "Hospital",
    "Bathroom",
    "Market",
    "Bus stop",
    "Today",
    "Tomorrow",
    "Yesterday",
    "Now",
    "Later",
    "Time",
    "Go",
    "Come",
    "Eat",
    "Drink",
    "Sit",
    "Stop",
]

# INCLUDE label variants that should count as a target word
ALIASES = {
    "you (plural)": "You",
    "how are you": "How",  # partial overlap — still useful signal; listed separately in notes
    "good morning": "Good",
    "good afternoon": "Good",
    "good evening": "Good",
    "good night": "Good",
}


def clean_label(raw: str) -> str:
    m = re.match(r"^\s*\d+\.\s*(.+)\s*$", raw)
    return (m.group(1) if m else raw).strip()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def load_include_labels(landmarks_dir: Path) -> Counter:
    counts: Counter = Counter()
    for meta_path in sorted(landmarks_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        label = clean_label(meta.get("label", ""))
        if label:
            counts[label] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--landmarks",
        type=Path,
        default=Path(__file__).resolve().parent / "landmarks",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "vocab_coverage.json",
    )
    parser.add_argument(
        "--vocab-file",
        type=Path,
        default=None,
        help="Optional text file with one target word per line (overrides default 50)",
    )
    args = parser.parse_args()

    if not args.landmarks.exists():
        print(f"ERROR: landmarks dir missing: {args.landmarks}")
        return 1

    if args.vocab_file:
        target = [
            ln.strip()
            for ln in args.vocab_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    else:
        target = list(DEFAULT_50_WORDS)

    include_counts = load_include_labels(args.landmarks)
    include_by_norm = {norm(k): k for k in include_counts}

    # Also map aliases from INCLUDE side
    for inc_label, target_name in ALIASES.items():
        if norm(inc_label) in include_by_norm and norm(target_name) not in {
            norm(t) for t in target
        }:
            pass  # aliases only used when matching targets

    have = []
    missing = []
    matched_include = {}

    for word in target:
        n = norm(word)
        hit = None
        if n in include_by_norm:
            hit = include_by_norm[n]
        else:
            # alias: if any INCLUDE label aliases to this target
            for inc_norm, canon in ((norm(a), norm(b)) for a, b in ALIASES.items()):
                if canon == n and inc_norm in include_by_norm:
                    hit = include_by_norm[inc_norm]
                    break
        if hit is not None:
            have.append(word)
            matched_include[word] = {
                "include_label": hit,
                "clips": int(include_counts[hit]),
            }
        else:
            missing.append(word)

    # Extra INCLUDE labels not in the 50-word list (bonus coverage)
    target_norms = {norm(w) for w in target}
    alias_norms = {norm(a) for a in ALIASES}
    extra = sorted(
        lab
        for lab in include_counts
        if norm(lab) not in target_norms and norm(lab) not in alias_norms
    )

    print(f"Target words: {len(target)}")
    print(f"INCLUDE labels on disk: {len(include_counts)} ({sum(include_counts.values())} clips)")
    print()
    print(f"HAVE ({len(have)}/{len(target)}):")
    for w in have:
        info = matched_include[w]
        print(f"  ✓ {w:15s}  <- INCLUDE '{info['include_label']}' ({info['clips']} clips)")
    print()
    print(f"MISSING ({len(missing)}/{len(target)}):")
    for w in missing:
        print(f"  ✗ {w}")
    print()
    print(f"EXTRA INCLUDE labels not in target list ({len(extra)}):")
    print(" ", ", ".join(extra[:40]) + (" ..." if len(extra) > 40 else ""))

    payload = {
        "target_count": len(target),
        "have_count": len(have),
        "missing_count": len(missing),
        "have": have,
        "missing": missing,
        "matched_include": matched_include,
        "extra_include_labels": extra,
        "include_label_counts": dict(sorted(include_counts.items())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
