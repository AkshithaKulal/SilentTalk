#!/usr/bin/env python3
"""Train a simple ISL gloss classifier on MediaPipe landmark sequences.

Uses pooled temporal stats (mean/std/min/max) + sklearn MLP — no GPU required.

Example:
  python train_classifier.py --landmarks .\\landmarks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_label(raw: str) -> str:
    m = re.match(r"^\s*\d+\.\s*(.+)\s*$", raw)
    return (m.group(1) if m else raw).strip()


FEAT_DIM_BASE = 225  # pose33*3 + left_hand21*3 + right_hand21*3


def sequence_to_features(seq: np.ndarray) -> np.ndarray:
    """seq: (T, D) -> fixed-length feature vector via temporal pooling (900 dims)."""
    if seq.ndim != 2 or seq.shape[0] == 0:
        return np.zeros(225 * 4, dtype=np.float32)
    mean = seq.mean(axis=0)
    std  = seq.std(axis=0)
    mn   = seq.min(axis=0)
    mx   = seq.max(axis=0)
    return np.concatenate([mean, std, mn, mx]).astype(np.float32)


def _feature_dim() -> int:
    return 225 * 4  # 900


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Generate augmented copies of a landmark sequence.

    Techniques used (all geometrically realistic for sign language):
    1. Gaussian jitter      — simulates natural hand tremor / sensor noise
    2. Temporal stretch     — simulates signing faster or slower
    3. Horizontal flip      — simulates left-handed signer (mirror x coords)
    4. Spatial scale        — simulates signing closer / further from camera

    Returns 3 augmented sequences (in addition to the original).
    """
    T, D = seq.shape
    aug = []

    # 1. Gaussian jitter — small noise on all landmarks
    noise_scale = 0.008
    jitter = seq + rng.normal(0, noise_scale, seq.shape).astype(np.float32)
    aug.append(jitter)

    # 2. Temporal stretch — resample to 85-115% of original length
    stretch_factor = rng.uniform(0.85, 1.15)
    new_T = max(3, int(T * stretch_factor))
    old_idx = np.linspace(0, T - 1, new_T)
    stretched = np.stack([
        np.interp(old_idx, np.arange(T), seq[:, d])
        for d in range(D)
    ], axis=1).astype(np.float32)
    aug.append(stretched)

    # 3. Horizontal flip — negate all x coords (every 3rd value starting at 0)
    # Layout: pose(33*3) | left_hand(21*3) | right_hand(21*3) — x at cols 0,3,6,...
    flipped = seq.copy()
    for start in range(0, D, 3):      # x coordinate of each landmark
        flipped[:, start] = 1.0 - seq[:, start]   # mirror: x → 1-x (normalized 0-1)
    aug.append(flipped)

    # 4. Spatial scale — scale all coords by 0.92-1.08 around centroid
    scale = rng.uniform(0.92, 1.08)
    centroid = seq.mean(axis=0)
    scaled = centroid + (seq - centroid) * scale
    aug.append(scaled.astype(np.float32))

    return aug  # 4 augmented versions


def load_dataset(landmarks_dir: Path, min_per_class: int, augment: bool = True,
                  aug_seed: int = 42):
    """Load landmarks, optionally augment, return feature matrix."""
    raw_seqs: list[np.ndarray] = []
    y_list:   list[str]        = []

    for npy in sorted(landmarks_dir.glob("*.npy")):
        meta_path = npy.with_suffix(".json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        label = clean_label(meta.get("label", ""))
        if not label:
            continue
        seq = np.load(npy)
        if seq.ndim != 2 or seq.shape[0] == 0:
            continue
        raw_seqs.append(seq)
        y_list.append(label)

    if not raw_seqs:
        raise RuntimeError(f"No samples found in {landmarks_dir}")

    all_counts = Counter(y_list)
    keep = {lab for lab, n in all_counts.items() if n >= min_per_class}
    dropped = sorted(lab for lab in all_counts if lab not in keep)
    if dropped:
        print(f"Dropping {len(dropped)} labels with <{min_per_class} samples: {dropped[:20]}"
              f"{'...' if len(dropped) > 20 else ''}")

    # Filter to kept labels
    kept_seqs  = [s for s, y in zip(raw_seqs, y_list) if y in keep]
    kept_labels = [y for y in y_list if y in keep]

    # Augmentation — adds 4× synthetic copies of every sample
    if augment:
        rng = np.random.default_rng(aug_seed)
        aug_seqs, aug_labels = [], []
        for seq, label in zip(kept_seqs, kept_labels):
            for aug_seq in augment_sequence(seq, rng):
                aug_seqs.append(aug_seq)
                aug_labels.append(label)
        all_seqs   = kept_seqs  + aug_seqs
        all_labels = kept_labels + aug_labels
        print(f"Augmentation: {len(kept_seqs)} original → {len(all_seqs)} total "
              f"({len(aug_seqs)} synthetic added)")
    else:
        all_seqs, all_labels = kept_seqs, kept_labels

    # Convert sequences to feature vectors
    X = np.stack([sequence_to_features(s) for s in all_seqs], axis=0)
    return X, np.array(all_labels), [], all_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Train INCLUDE landmark classifier")
    parser.add_argument(
        "--landmarks",
        type=Path,
        default=Path(__file__).resolve().parent / "landmarks",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true", default=False,
                        help="Disable data augmentation (enabled by default)")
    args = parser.parse_args()

    if not args.landmarks.exists():
        print(f"ERROR: landmarks dir missing: {args.landmarks}", file=sys.stderr)
        return 1

    print(f"Loading from {args.landmarks} ...")
    augment = not args.no_augment
    X, y_raw, _paths, all_counts = load_dataset(
        args.landmarks, args.min_per_class, augment=augment, aug_seed=args.seed)
    print(f"Usable samples={len(y_raw)} classes={len(set(y_raw))} feat_dim={X.shape[1]}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Stratified split needs >=2 samples per class in both splits ideally
    class_counts = Counter(y_raw)
    rare = [lab for lab, n in class_counts.items() if n < 2]
    if rare:
        print(f"WARNING: {len(rare)} classes have only 1 sample after filter; excluding them from split")
        mask = np.array([lab not in set(rare) for lab in y_raw])
        X, y_raw = X[mask], y_raw[mask]
        y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(512, 256),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=64,
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=args.seed,
                    verbose=False,
                ),
            ),
        ]
    )

    print("Training MLP ...")
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(y_test, pred)
    top3 = top_k_accuracy_score(y_test, proba, k=min(3, proba.shape[1]))
    top5 = top_k_accuracy_score(y_test, proba, k=min(5, proba.shape[1]))

    report = classification_report(
        y_test,
        pred,
        target_names=list(le.classes_),
        zero_division=0,
    )

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Top-3:    {top3:.4f}")
    print(f"Top-5:    {top5:.4f}")
    print("\nClassification report (test set):")
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "sign_classifier.joblib"
    encoder_path = args.out_dir / "label_encoder.joblib"
    report_path = args.out_dir / "train_report.json"

    joblib.dump(clf, model_path)
    joblib.dump(le, encoder_path)

    summary = {
        "samples_total_on_disk": int(sum(all_counts.values())),
        "samples_used": int(len(y_raw)),
        "num_classes": int(len(le.classes_)),
        "classes": list(le.classes_),
        "accuracy": float(acc),
        "top3": float(top3),
        "top5": float(top5),
        "feature_dim": int(X.shape[1]),
        "model_path": str(model_path),
    }
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    print(f"\nSaved {model_path}")
    print(f"Saved {encoder_path}")
    print(f"Saved {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
