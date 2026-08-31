#!/usr/bin/env python3
"""Quick GPU check before overnight training."""

from __future__ import annotations

import sys
from pathlib import Path

ISL = Path(__file__).resolve().parent
sys.path.insert(0, str(ISL))

from torch_device import cuda_ready, gpu_summary  # noqa: E402


def main() -> int:
    print(gpu_summary())
    if cuda_ready():
        import torch

        print(f"OK — training will use: {torch.cuda.get_device_name(0)}")
        return 0
    print(
        "CUDA not available.\n"
        "  Windows: .\\scripts\\install_torch_cuda.ps1\n"
        "  Or:      python setup.py",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
