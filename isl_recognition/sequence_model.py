"""BiLSTM sign classifier used by train_sequence.py and (once trained) live predict."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FEAT_DIM = 225  # pose33*3 + left21*3 + right21*3
SEQ_LEN = 64


class SignBiLSTM(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 256, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            FEAT_DIM,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 225)
        out, (h, _) = self.lstm(x)
        # last layer forward + backward hidden
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(h_cat)


class SignBiLSTMAttn(nn.Module):
    """V2: attention over the full sequence (hands + motion), not last step only."""

    def __init__(self, num_classes: int, hidden: int = 384, layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            FEAT_DIM,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        pooled = (out * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled)


def build_model(num_classes: int, ckpt: dict | None = None, arch: str = "bilstm", hidden: int = 256) -> nn.Module:
    if ckpt is not None:
        arch = ckpt.get("arch", arch)
        hidden = int(ckpt.get("hidden", hidden))
        layers = int(ckpt.get("layers", 2))
        dropout = float(ckpt.get("dropout", 0.3 if arch == "bilstm" else 0.4))
    else:
        layers = 2
        dropout = 0.3 if arch == "bilstm" else 0.4
    if arch in ("bilstm_attn", "v2"):
        return SignBiLSTMAttn(num_classes, hidden=hidden, layers=layers, dropout=dropout)
    return SignBiLSTM(num_classes, hidden=hidden, layers=layers, dropout=dropout)


def resample(seq: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    if seq.ndim != 2 or seq.shape[0] == 0:
        return np.zeros((length, FEAT_DIM), dtype=np.float32)
    if seq.shape[0] == length:
        return seq.astype(np.float32, copy=False)
    old = np.linspace(0.0, 1.0, seq.shape[0])
    new = np.linspace(0.0, 1.0, length)
    cols = [
        np.interp(new, old, seq[:, d]).astype(np.float32)
        for d in range(seq.shape[1])
    ]
    return np.stack(cols, axis=1)


def normalize_skeleton(seq: np.ndarray) -> np.ndarray:
    """Translate/scale pose by hips+shoulders; hands by wrist. Fixes camera distance."""
    if seq.ndim != 2 or seq.shape[1] < FEAT_DIM or seq.shape[0] == 0:
        return seq.astype(np.float32, copy=False)
    out = seq.astype(np.float32, copy=True)
    t = out.shape[0]
    pose = out[:, :99].reshape(t, 33, 3)
    left = out[:, 99:162].reshape(t, 21, 3)
    right = out[:, 162:225].reshape(t, 21, 3)

    mid = (pose[:, 23] + pose[:, 24]) * 0.5
    shoulder = np.linalg.norm(pose[:, 11] - pose[:, 12], axis=-1, keepdims=True)
    shoulder = np.clip(shoulder, 1e-3, None)
    pose = (pose - mid[:, None, :]) / shoulder[:, None, :]

    def hand_rel(hand: np.ndarray) -> np.ndarray:
        present = np.abs(hand).sum(axis=(1, 2)) > 1e-6
        if not present.any():
            return hand
        wrist = hand[:, 0:1, :]
        span = np.linalg.norm(hand[:, 5] - hand[:, 17], axis=-1, keepdims=True)
        span = np.clip(span, 1e-3, None)
        rel = (hand - wrist) / span[:, None, :]
        rel[~present] = 0
        return rel

    left = hand_rel(left)
    right = hand_rel(right)
    out[:, :99] = pose.reshape(t, 99)
    out[:, 99:162] = left.reshape(t, 63)
    out[:, 162:225] = right.reshape(t, 63)
    return out


def hflip_swap_hands(seq: np.ndarray) -> np.ndarray:
    """Mirror x and swap left/right hands so flip is a real left-handed signer."""
    out = seq.copy()
    out[:, 0::3] *= -1.0
    left = out[:, 99:162].copy()
    right = out[:, 162:225].copy()
    out[:, 99:162] = right
    out[:, 162:225] = left
    return out


def prepare_clip(seq: np.ndarray, augment: bool = False, rng: np.random.Generator | None = None, strong: bool = False) -> np.ndarray:
    seq = normalize_skeleton(seq)
    if augment and rng is not None:
        noise = 0.018 if strong else 0.01
        if rng.random() < 0.65:
            seq = seq + rng.normal(0, noise, seq.shape).astype(np.float32)
        if rng.random() < 0.55 and seq.shape[0] >= 4:
            factor = float(rng.uniform(0.75, 1.25) if strong else rng.uniform(0.85, 1.15))
            new_t = max(4, int(seq.shape[0] * factor))
            seq = resample(seq, new_t)
        if rng.random() < 0.5:
            seq = hflip_swap_hands(seq)
        if strong and rng.random() < 0.35 and seq.shape[0] >= 8:
            t0 = int(rng.integers(0, seq.shape[0] - 3))
            span = int(rng.integers(2, max(3, seq.shape[0] // 6)))
            seq[t0:t0 + span] = 0
    return resample(seq, SEQ_LEN)


def load_bundle(path: Path, device: torch.device | None = None):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(len(classes), ckpt=ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, classes, device


@torch.no_grad()
def predict_topk(model: nn.Module, classes: list[str], seq: np.ndarray, device: torch.device, k: int = 5):
    x = prepare_clip(seq, augment=False)
    t = torch.from_numpy(x).unsqueeze(0).to(device)
    logits = model(t)[0]
    prob = torch.softmax(logits, dim=0).cpu().numpy()
    idx = np.argsort(prob)[::-1][:k]
    return [(classes[i], float(prob[i])) for i in idx]


def hand_activity(seq: np.ndarray) -> float:
    """Fraction of frames with a non-empty hand (left or right)."""
    if seq.ndim != 2 or seq.shape[0] == 0 or seq.shape[1] < FEAT_DIM:
        return 0.0
    left = np.abs(seq[:, 99:162]).sum(axis=1)
    right = np.abs(seq[:, 162:225]).sum(axis=1)
    present = (left > 1e-4) | (right > 1e-4)
    return float(present.mean())


def save_bundle(path: Path, model: nn.Module, classes: list[str], extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "classes": classes,
        "feat_dim": FEAT_DIM,
        "seq_len": SEQ_LEN,
        **extra,
    }
    torch.save(payload, path)
    (path.with_suffix(".classes.json")).write_text(
        json.dumps({"classes": classes, **{k: extra[k] for k in extra if k != "state_dict"}}, indent=2),
        encoding="utf-8",
    )
