"""Checkpoint adapter for the pinned pytorch/examples ImageNet recipe."""

from pathlib import Path
from typing import Any

import torch

NAME = "pytorch_examples_imagenet"


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.pth.tar"


def normalize(path: Path) -> object:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key: checkpoint[key]
        for key in ("epoch", "state_dict", "optimizer", "scheduler")
    }
