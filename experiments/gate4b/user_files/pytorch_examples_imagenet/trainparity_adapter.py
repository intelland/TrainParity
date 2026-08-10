"""Explicit ImageNet command and checkpoint semantics; no orchestration."""

import sys
from pathlib import Path
from typing import Any

import torch

from trainparity import ProcessExecutionPlan


class Case:
    name, split_step, total_step = "pytorch_examples_imagenet", 2, 3

    def command(self, plan: ProcessExecutionPlan) -> list[str]:
        launcher = "import os,runpy,sys;os.chdir(sys.argv.pop(1));runpy.run_path(sys.argv.pop(1),run_name='__main__')"
        command = [sys.executable, "-c", launcher, str(plan.run_dir), str(plan.cwd / "imagenet/main.py"), str(plan.cwd / ".trainparity_data"), "--arch", "shufflenet_v2_x0_5", "--workers", "0", "--epochs", str(plan.end_step), "--batch-size", "4", "--lr", "0.01", "--seed", "23", "--print-freq", "1000"]
        return command if plan.resume_from is None else [*command, "--resume", str(plan.resume_from)]

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "checkpoint.pth.tar"

    def observe_checkpoint(self, path: Path) -> dict[str, object]:
        value: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        return {key: value[key] for key in ("epoch", "state_dict", "optimizer", "scheduler")}
