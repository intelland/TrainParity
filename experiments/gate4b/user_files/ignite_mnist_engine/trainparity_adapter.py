"""Explicit Ignite command and checkpoint semantics; no orchestration."""

import os
import sys
from pathlib import Path
from typing import Any

import torch

from trainparity.api import ProcessExecutionPlan


class Case:
    name, split_step, total_step = "ignite_mnist_engine", 2, 4

    def command(self, plan: ProcessExecutionPlan) -> list[str]:
        driver = os.environ["TRAINPARITY_GATE4B_IGNITE_DRIVER"]
        return [sys.executable, driver, str(plan.cwd), str(plan.run_dir), str(plan.end_step), "" if plan.resume_from is None else str(plan.resume_from)]

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "checkpoint.pt"

    def observe_checkpoint(self, path: Path) -> dict[str, object]:
        value: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        observed = {key: value[key] for key in ("trainer", "model", "optimizer", "lr_scheduler")}
        observed["rng"] = {"torch_cpu": torch.load(path.with_name("rng_state.pt"), weights_only=True)}
        return observed
