"""Adapter for pytorch/examples ImageNet training at a pinned commit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


# ADAPTER LOGIC START
class ImageNetAdapter:
    name = "pytorch_examples_imagenet"
    repository = "https://github.com/pytorch/examples.git"
    commit = "acc295dc7b90714f1bf47f06004fc19a7fe235c4"
    license_id = "BSD-3-Clause"
    structure = "conventional image classifier"
    fault_name = "scheduler_last_epoch_off_by_one"
    split_step, total_step = 2, 3

    def prepare_command(self, checkout: Path, data_root: Path) -> list[str]:
        return ["python", "-m", "experiments.gate4.drivers.imagenet", "prepare", str(data_root)]

    def run_command(self, checkout: Path, data_root: Path, run_dir: Path, end_step: int, resume_from: Path | None) -> list[str]:
        command = ["python", "-m", "experiments.gate4.drivers.imagenet", "run", str(checkout), str(data_root), str(run_dir), str(end_step)]
        return command if resume_from is None else [*command, str(resume_from)]

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "checkpoint.pth.tar"

    def normalize_checkpoint(self, path: Path) -> object:
        value: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        return {key: value[key] for key in ("epoch", "state_dict", "optimizer", "scheduler")}

    def inject_fault(self, path: Path) -> None:
        value: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        value["scheduler"]["last_epoch"] -= 1
        torch.save(value, path)

    def handwritten_state(self, path: Path) -> object:
        return self.normalize_checkpoint(path)["state_dict"]  # type: ignore[index]
# ADAPTER LOGIC END
