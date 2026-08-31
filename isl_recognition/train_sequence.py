#!/usr/bin/env python3
"""Train INCLUDE sequence classifier (v1 BiLSTM or v2 attention + official splits).

v1 (default):
  python train_sequence.py --landmarks .\\landmarks --out .\\transfer_pack

v2 (office production):
  python train_sequence.py --v2 --landmarks .\\landmarks --out .\\transfer_pack --require-gpu
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

from sequence_model import build_model, prepare_clip, save_bundle
from torch_device import configure_for_training, gpu_summary, resolve_device

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
    def __init__(
        self,
        items: list[tuple[Path, str]],
        class_to_idx: dict[str, int],
        augment: bool,
        strong: bool = False,
    ):
        self.items = items
        self.class_to_idx = class_to_idx
        self.augment = augment
        self.strong = strong

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        seq = np.load(path)
        rng = np.random.default_rng() if self.augment else None
        x = prepare_clip(seq, augment=self.augment, rng=rng, strong=self.strong)
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


def class_weight_tensor(classes: list[str], counts: Counter, device) -> torch.Tensor:
    arr = np.array([counts[c] for c in classes], dtype=np.float64)
    w = 1.0 / np.clip(arr, 1.0, None)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train INCLUDE sequence model")
    parser.add_argument("--landmarks", type=Path, default=Path(__file__).resolve().parent / "landmarks")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "transfer_pack")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--v2", action="store_true", help="Attention BiLSTM + official splits + class weights")
    parser.add_argument("--no-official-split", action="store_true", help="Even in v2, use random 80/20")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "gpu"),
        default="auto",
        help="Training device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit if CUDA is not available (recommended on office PC)",
    )
    args = parser.parse_args()

    v2 = args.v2
    epochs = args.epochs if args.epochs is not None else (80 if v2 else 40)
    batch = args.batch if args.batch is not None else (48 if v2 else 32)
    lr = args.lr if args.lr is not None else (8e-4 if v2 else 1e-3)
    patience = args.patience if args.patience is not None else (12 if v2 else 8)
    arch = "bilstm_attn" if v2 else "bilstm"
    hidden = 384 if v2 else 256

    if not args.landmarks.exists():
        print(f"ERROR: landmarks folder not found: {args.landmarks}", file=sys.stderr)
        return 1

    items = load_items(args.landmarks)
    if not items:
        print("ERROR: no landmark .npy files with labels", file=sys.stderr)
        return 1

    counts = Counter(lab for _, lab in items)
    print(f"clips={len(items)}  classes={len(counts)}  mode={'v2' if v2 else 'v1'}")
    if len(counts) < 50:
        print(
            "WARNING: fewer than 50 classes. Full INCLUDE extract is not done yet.",
            file=sys.stderr,
        )

    split_name = "stratified 80/20 by word, before augment"
    split_stats = {}
    if v2 and not args.no_official_split:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "include_official"))
        from official_splits import split_items as official_split_items

        cache = Path(__file__).resolve().parent / "include_official" / "splits"
        train_items, test_items, split_stats = official_split_items(items, cache)
        print(
            f"official split match_rate={split_stats['match_rate']:.3f}  "
            f"train={split_stats['matched_train']} test={split_stats['matched_test']} "
            f"unmatched_added_to_train={split_stats['unmatched']}"
        )
        if split_stats["matched_test"] < 100 or split_stats["match_rate"] < 0.5:
            print("Official match too low — falling back to stratified 80/20", file=sys.stderr)
            train_items, test_items = stratified_split(items)
            split_name = "stratified 80/20 fallback (official match low)"
        else:
            split_name = "official INCLUDE train/test (+ unmatched in train)"
    else:
        train_items, test_items = stratified_split(items)

    classes = sorted(counts)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"train={len(train_items)}  test={len(test_items)}  ({split_name})")
    print(gpu_summary())

    device = resolve_device(args.device, require_gpu=args.require_gpu)
    configure_for_training(device)
    print(f"training on device={device}  arch={arch}  hidden={hidden}")

    train_ds = ClipDataset(train_items, class_to_idx, augment=True, strong=v2)
    test_ds = ClipDataset(test_items, class_to_idx, augment=False, strong=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(len(classes), arch=arch, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4 if v2 else 1e-4)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
        if v2
        else None
    )
    if v2:
        weights = class_weight_tensor(classes, counts, device)
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.08)
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
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
        if scheduler is not None:
            scheduler.step()
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
            if stale >= patience:
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
        "version": "v2" if v2 else "v1",
        "arch": arch,
        "hidden": hidden,
        "layers": 2,
        "dropout": 0.4 if v2 else 0.3,
        "split": split_name,
        "split_stats": split_stats,
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
