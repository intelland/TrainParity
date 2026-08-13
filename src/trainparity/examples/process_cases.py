"""Tiny command-oriented cases used to contract-test the production runner."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from trainparity.protocols import ProcessExecutionPlan


def _train(
    run_dir: Path, end_step: int, resume: Path | None, mode: str, nudge: float
) -> None:
    if mode == "error":
        raise RuntimeError("intentional child failure")
    if mode == "slow":
        time.sleep(1.0)
    if (
        os.environ.get("TRAINPARITY_REQUIRE_TORCH_FORCE") == "1"
        and os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") != "1"
    ):
        raise RuntimeError("required environment was not propagated")
    if resume is None:
        torch.manual_seed(29)
        state: dict[str, Any] = {
            "step": 0,
            "model": {"weight": torch.tensor([1.0])},
            "optimizer": {"momentum": torch.tensor([0.0])},
            "scheduler": {"last_epoch": 0},
        }
    else:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        torch.set_rng_state(state["rng"]["torch_cpu"])
    while state["step"] < end_step:
        sample = torch.rand(1)
        state["optimizer"]["momentum"] = 0.9 * state["optimizer"]["momentum"] + sample
        state["model"]["weight"] += state["optimizer"]["momentum"]
        state["scheduler"]["last_epoch"] += 1
        state["step"] += 1
    if nudge:
        state["model"]["weight"] += nudge
    state["rng"] = {"torch_cpu": torch.get_rng_state()}
    if mode == "nondeterministic":
        state["process_nonce"] = os.getpid()
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, run_dir / "checkpoint.pt")


class DeterministicProcessCase:
    """Small exact process case covering model, optimizer, scheduler, and RNG."""

    name = "deterministic_process"
    split_step = 2
    total_step = 4
    mode = "clean"

    def command(self, plan: ProcessExecutionPlan) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "trainparity.examples.process_cases",
            "train",
            "--run-dir",
            str(plan.run_dir),
            "--end-step",
            str(plan.end_step),
            "--mode",
            self.mode,
        ]
        if plan.resume_from is not None:
            command.extend(("--resume", str(plan.resume_from)))
        return command

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "checkpoint.pt"

    def observe_checkpoint(self, path: Path) -> dict[str, object]:
        state: dict[str, object] = torch.load(path, map_location="cpu", weights_only=True)
        return {
            key: state[key]
            for key in ("model", "optimizer", "scheduler", "rng")
        }


class NondeterministicProcessCase(DeterministicProcessCase):
    mode = "nondeterministic"

    def observe_checkpoint(self, path: Path) -> dict[str, object]:
        state: dict[str, object] = torch.load(path, map_location="cpu", weights_only=True)
        observed = super().observe_checkpoint(path)
        observed["process_nonce"] = state["process_nonce"]
        return observed


class BaselineToleranceProcessCase(DeterministicProcessCase):
    """Baseline B differs by a declared small floating-point perturbation."""

    baseline_nudge = 1e-5

    def command(self, plan: ProcessExecutionPlan) -> list[str]:
        command = super().command(plan)
        if plan.phase == "baseline_b":
            command.extend(("--nudge", str(self.baseline_nudge)))
        return command


class BaselineOutsideToleranceProcessCase(BaselineToleranceProcessCase):
    """Baseline B differs beyond the tolerance used by the regression test."""

    baseline_nudge = 1e-2


class ErrorProcessCase(DeterministicProcessCase):
    mode = "error"


class SlowProcessCase(DeterministicProcessCase):
    mode = "slow"


class CommandCallbackErrorCase(DeterministicProcessCase):
    def command(self, plan: ProcessExecutionPlan) -> list[str]:
        raise RuntimeError("intentional command callback failure")


class SystemExitCommandCase(DeterministicProcessCase):
    def command(self, plan: ProcessExecutionPlan) -> list[str]:
        raise SystemExit(7)


class PostRunCheckpointPathErrorCase(DeterministicProcessCase):
    def checkpoint_path(self, run_dir: Path) -> Path:
        raise FileNotFoundError("intentional post-run checkpoint lookup failure")


class CandidateResumeCheckpointPathErrorCase(DeterministicProcessCase):
    def checkpoint_path(self, run_dir: Path) -> Path:
        if run_dir.name == "candidate_resume":
            raise FileNotFoundError("intentional staging location failure")
        return super().checkpoint_path(run_dir)


class ObservationCallbackErrorCase(DeterministicProcessCase):
    def observe_checkpoint(self, path: Path) -> dict[str, object]:
        raise RuntimeError("intentional observation callback failure")


class UnsupportedProcessCase(DeterministicProcessCase):
    def observe_checkpoint(self, path: Path) -> dict[str, object]:
        observed = super().observe_checkpoint(path)
        observed["unsupported"] = object()
        return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train",))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--nudge", type=float, default=0.0)
    args = parser.parse_args()
    _train(args.run_dir, args.end_step, args.resume, args.mode, args.nudge)


if __name__ == "__main__":
    main()
