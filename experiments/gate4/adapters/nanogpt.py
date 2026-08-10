"""Adapter for karpathy/nanoGPT at a pinned commit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

# ADAPTER LOGIC START
class NanoGptAdapter:
    name = "nanogpt"
    repository = "https://github.com/karpathy/nanoGPT.git"
    commit = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
    license_id = "MIT"
    structure = "small language model"
    fault_name = "resume_iteration_off_by_one"
    split_step, total_step = 2, 4

    def prepare_command(self, checkout: Path, data_root: Path) -> list[str]:
        return ["python", "-m", "experiments.gate4.drivers.nanogpt", "prepare", str(checkout)]

    def run_command(self, checkout: Path, data_root: Path, run_dir: Path, end_step: int, resume_from: Path | None) -> list[str]:
        command = ["python", "-m", "experiments.gate4.drivers.nanogpt", "run", str(checkout), str(run_dir), str(end_step)]
        return command if resume_from is None else [*command, "resume"]

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "ckpt.pt"

    def normalize_checkpoint(self, path: Path) -> object:
        value: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        return {key: value[key] for key in ("model", "optimizer", "model_args", "iter_num", "best_val_loss")}

    def inject_fault(self, path: Path) -> None:
        value: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        value["iter_num"] -= 1
        torch.save(value, path)

    def handwritten_state(self, path: Path) -> object:
        return self.normalize_checkpoint(path)["model"]  # type: ignore[index]
# ADAPTER LOGIC END

