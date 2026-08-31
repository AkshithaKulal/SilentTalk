#!/usr/bin/env python3
"""Train a NEW INCLUDE sequence classifier from scratch (BiLSTM).

This does NOT load or continue the old 84-class sklearn MLP.
Run it on the office PC after:

  1. include_prepare.py  (full zips extracted)
  2. audit shows ~4292 videos / 263 words / 15 categories
  3. extract_landmarks.py on F:\\include_dataset\\extracted

Example:

  python train_sequence.py --landmarks .\\landmarks --out .\\transfer_pack

Writes sign_bilstm.pt (the live app prefers this over sign_classifier.joblib).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from sequence_model import SignBiLSTM, prepare_clip, save_bundle

NUM_RE = re.compile(r"^\s*\d+\.\s*(.+)\s*$")


def clean_label(raw: str) -> str:
    raw = (raw or "").strip()
    m = NUM_RE.match(raw)
    return (m.group(1) if m else raw).strip()


def label_from_stem(stem: str) -> str:
    parts = stem.split("__")
    if len(parts) >= 3 and parts[-2].lower() == "extra":
        return clean_label(parts[-3])
    for part in reversed(parts):
        if part.lower() == "extra":
            continue
        if NUM_RE.match(part):
            return clean_label(part)
    if len(parts) >= 2:
        return clean_label(parts[-2])
    return clean_label(parts[0] if parts else "")


def load_items(land_dir: Path) -> list[tuple[Path, str]]:
    items = []
    for npy in sorted(land_dir.glob("*.npy")):
        meta_path = npy.with_suffix(".json")
        label = ""
        if meta_path.exists():
            try:
                label = json.loads(meta_path.read_text(encoding="utf-8")).get("label") or ""
            except json.JSONDecodeError:
                label = ""
        label = clean_label(label) if label else label_from_stem(npy.stem)
        if not label or label.lower() == "extra":
            continue
        items.append((npy, label))
    return items


class ClipDataset(Dataset):
    def __init__(self, items: list[tuple[Path, str]], class_to_idx: dict[str, int], augment: bool):
        self.items = items
        self.class_to_idx = class_to_idx
        self.augment = augment
        self.rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        seq = np.load(path)
        rng = np.random.default_rng() if self.augment else None
        x = prepare_clip(seq, augment=self.augment, rng=rng)
        y = self.class_to_idx[label]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def stratified_split(items: list[tuple[Path, str]], seed: int = 42):
    labels = [lab for _, lab in items]
    counts = Counter(labels)
    rare = [lab for lab, n in counts.items() if n < 2]
    if rare:
        print(f"Classes with <2 clips (all go to train): {rare}")
    keep = [(p, lab) for p, lab in items if counts[lab] >= 2]
    hold = [(p, lab) for p, lab in items if counts[lab] < 2]
    y = [lab for _, lab in keep]
    train, test = train_test_split(keep, test_size=0.2, random_state=seed, stratify=y)
    train = train + hold
    return train, test


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct = top3 = top5 = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        k = min(5, logits.shape[1])
        topk = logits.topk(k, dim=1).indices
        top3 += int((topk[:, : min(3, k)] == y[:, None]).any(dim=1).sum().item())
        top5 += int((topk[:, :k] == y[:, None]).any(dim=1).sum().item())
        n += y.numel()
    return correct / n, top3 / n, top5 / n


def main() -> int:
    parser = argparse.ArgumentParser(description="Train INCLUDE BiLSTM from scratch")
    parser.add_argument("--landmarks", type=Path, default=Path(__file__).resolve().parent / "landmarks")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "transfer_pack")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    if not args.landmarks.exists():
        print(f"ERROR: landmarks folder not found: {args.landmarks}", file=sys.stderr)
        return 1

    items = load_items(args.landmarks)
    if not items:
        print("ERROR: no landmark .npy files with labels", file=sys.stderr)
        return 1

    counts = Counter(lab for _, lab in items)
    print(f"clips={len(items)}  classes={len(counts)}")
    if len(counts) < 50:
        print(
            "WARNING: fewer than 50 classes. Full INCLUDE extract is not done yet. "
            "Do not treat this run as the production 263-class model.",
            file=sys.stderr,
        )

    train_items, test_items = stratified_split(items)
    classes = sorted(counts)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"train={len(train_items)}  test={len(test_items)}  (split BEFORE any augment)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_ds = ClipDataset(train_items, class_to_idx, augment=True)
    test_ds = ClipDataset(test_items, class_to_idx, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = SignBiLSTM(num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item()) * y.numel()
            seen += y.numel()
        acc, top3, top5 = eval_epoch(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "test_acc": acc,
            "test_top3": top3,
            "test_top5": top5,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}  loss={row['train_loss']:.4f}  "
            f"acc={acc:.4f}  top3={top3:.4f}  top5={top5:.4f}"
        )
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    if best_state is None:
        print("ERROR: training produced no weights", file=sys.stderr)
        return 1
    model.load_state_dict(best_state)
    acc, top3, top5 = eval_epoch(model, test_loader, device)

    extra = {
        "num_classes": len(classes),
        "clips": len(items),
        "train_clips": len(train_items),
        "test_clips": len(test_items),
        "test_acc": acc,
        "test_top3": top3,
        "test_top5": top5,
        "from_scratch": True,
        "not_the_old_mlp": True,
        "split": "stratified 80/20 by word, before augment",
        "history": history,
    }
    out_path = args.out / "sign_bilstm.pt"
    save_bundle(out_path, model, classes, extra)
    report = {k: extra[k] for k in extra if k != "history"}
    report["classes"] = classes
    (args.out / "sequence_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nHeld-out  acc={acc:.4f}  top3={top3:.4f}  top5={top5:.4f}")
    print(f"Saved {out_path}")
    print("This is a NEW model. The old sign_classifier.joblib was not used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
