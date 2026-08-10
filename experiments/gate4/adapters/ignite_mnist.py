"""Adapter for the PyTorch-Ignite save/resume Engine example."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

# ADAPTER LOGIC START
class IgniteMnistAdapter:
    name = "ignite_mnist_engine"
    repository = "https://github.com/pytorch/ignite.git"
    commit = "e08ff9257ed18d8d805304e32ba85a44553195fc"
    license_id = "BSD-3-Clause"
    structure = "trainer engine with scheduler and metric state"
    fault_name = "scheduler_last_epoch_off_by_one"
    split_step, total_step = 2, 4

    def prepare_command(self, checkout: Path, data_root: Path) -> list[str]:
        return ["python", "-m", "experiments.gate4.drivers.ignite_mnist", "prepare"]

    def run_command(self, checkout: Path, data_root: Path, run_dir: Path, end_step: int, resume_from: Path | None) -> list[str]:
        command = ["python", "-m", "experiments.gate4.drivers.ignite_mnist", "run", str(checkout), str(run_dir), str(end_step)]
        return command if resume_from is None else [*command, str(resume_from)]

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "checkpoint.pt"

    def normalize_checkpoint(self, path: Path) -> object:
        value: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        return {key: value[key] for key in ("trainer", "model", "optimizer", "lr_scheduler", "train_running_loss")}

    def inject_fault(self, path: Path) -> None:
        value: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        value["lr_scheduler"]["last_epoch"] -= 1
        torch.save(value, path)

    def handwritten_state(self, path: Path) -> object:
        return self.normalize_checkpoint(path)["model"]  # type: ignore[index]
# ADAPTER LOGIC END
