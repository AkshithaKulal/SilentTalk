"""PyTorch device helpers — shared by train_sequence and office_overnight."""

from __future__ import annotations

import sys


def cuda_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def gpu_summary() -> str:
    try:
        import torch
    except ImportError:
        return "torch not installed"

    lines = [f"torch {torch.__version__}", f"cuda.is_available()={torch.cuda.is_available()}"]
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_memory / (1024**3)
        lines.append(f"gpu={props.name}  vram={mem_gb:.1f} GB")
    return "  |  ".join(lines)


def resolve_device(prefer: str = "auto", require_gpu: bool = False):
    import torch

    prefer = (prefer or "auto").lower()
    if prefer == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif prefer in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            msg = (
                "CUDA requested but not available.\n"
                f"  {gpu_summary()}\n"
                f"  Fix on office PC:\n"
                f"    .\\scripts\\install_torch_cuda.ps1\n"
                f"  Or: python setup.py  (Step 3 installs torch+cu121)\n"
                f"  Then verify:\n"
                f'    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
            )
            if require_gpu:
                print(msg, file=sys.stderr)
                raise SystemExit(1)
            print(msg, file=sys.stderr)
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif prefer == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"unknown device: {prefer}")

    if require_gpu and device.type != "cuda":
        print(
            f"ERROR: --require-gpu but using {device}.\n  {gpu_summary()}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return device


def configure_for_training(device) -> None:
    import torch

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
