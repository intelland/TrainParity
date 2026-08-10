"""Checkpoint adapter for the pinned nanoGPT recipe."""

from pathlib import Path
from typing import Any

import torch

NAME = "nanogpt"


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "ckpt.pt"


def normalize(path: Path) -> object:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key: checkpoint[key]
        for key in ("model", "optimizer", "model_args", "iter_num", "best_val_loss")
    }
