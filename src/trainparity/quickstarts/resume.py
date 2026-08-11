"""Run clean and intentionally faulty command-oriented CPU resume checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from trainparity.api import (
    MACHINE_REPORT_SCHEMA_VERSION,
    PACKAGE_VERSION,
    ProcessExecutionPlan,
    check_resume,
)


def _train(run_dir: Path, end_step: int, resume_from: Path | None, fault: bool) -> None:
    if resume_from is None:
        torch.manual_seed(37)
        state: dict[str, Any] = {
            "step": 0,
            "model": {"weight": torch.tensor([1.0])},
            "optimizer": {"momentum": torch.tensor([0.0])},
            "scheduler": {"last_epoch": 0},
        }
    else:
        state = torch.load(resume_from, map_location="cpu", weights_only=True)
        torch.set_rng_state(state["rng"]["torch_cpu"])
        if fault:
            state["scheduler"]["last_epoch"] = 0
    while state["step"] < end_step:
        sample = torch.rand(1)
        state["optimizer"]["momentum"] = 0.8 * state["optimizer"]["momentum"] + sample
        state["model"]["weight"] += state["optimizer"]["momentum"]
        state["scheduler"]["last_epoch"] += 1
        state["step"] += 1
    state["rng"] = {"torch_cpu": torch.get_rng_state()}
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, run_dir / "checkpoint.pt")


class CleanCase:
    """A complete model/optimizer/scheduler/RNG resume adapter."""

    name = "quickstart_resume_clean"
    split_step = 2
    total_step = 4
    fault = False

    def command(self, plan: ProcessExecutionPlan) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "trainparity.quickstarts.resume",
            "worker",
            "--run-dir",
            str(plan.run_dir),
            "--end-step",
            str(plan.end_step),
        ]
        if plan.resume_from is not None:
            command.extend(("--resume-from", str(plan.resume_from)))
        if self.fault:
            command.append("--fault")
        return command

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "checkpoint.pt"

    def observe_checkpoint(self, path: Path) -> dict[str, object]:
        state: dict[str, object] = torch.load(path, map_location="cpu", weights_only=True)
        return {key: state[key] for key in ("model", "optimizer", "scheduler", "rng")}


class FaultyCase(CleanCase):
    """An adapter whose resume path intentionally resets scheduler state."""

    name = "quickstart_resume_fault"
    fault = True


def run() -> dict[str, object]:
    """Return a clean PASS and an intentional scheduler-state FAIL."""
    clean = check_resume("trainparity.quickstarts.resume:CleanCase")
    fault = check_resume("trainparity.quickstarts.resume:FaultyCase")
    return {
        "schema_version": MACHINE_REPORT_SCHEMA_VERSION,
        "trainparity_version": PACKAGE_VERSION,
        "clean": clean.to_dict(),
        "intentional_fail": fault.to_dict(),
    }


def main() -> int:
    """Run a worker command or print both example outcomes as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", choices=("worker",))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--fault", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "worker":
        if arguments.run_dir is None or arguments.end_step is None:
            parser.error("worker requires --run-dir and --end-step")
        _train(
            arguments.run_dir,
            arguments.end_step,
            arguments.resume_from,
            arguments.fault,
        )
        return 0
    payload = run()
    print(json.dumps(payload, indent=2, sort_keys=True))
    clean = payload["clean"]
    fault = payload["intentional_fail"]
    assert isinstance(clean, dict) and isinstance(fault, dict)
    return 0 if clean["outcome"] == "PASS" and fault["outcome"] == "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
