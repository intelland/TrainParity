"""Checkpoint adapter for the pinned Ignite MNIST engine recipe."""

from pathlib import Path
from typing import Any

import torch

NAME = "ignite_mnist_engine"


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.pt"


def normalize(path: Path) -> object:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    normalized = {
        key: checkpoint[key]
        for key in ("trainer", "model", "optimizer", "lr_scheduler")
    }
    normalized["rng"] = {"torch_cpu": torch.load(path.with_name("rng_state.pt"), weights_only=True)}
    return normalized
