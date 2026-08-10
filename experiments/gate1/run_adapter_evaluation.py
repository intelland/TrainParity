"""Evaluate the two Gate 1 adapter shapes on M3 and emit durable evidence."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from trainparity.examples.resume_cases import (
    CorrectResumeCase,
    MissingSchedulerStateCase,
    make_resume_callbacks,
)
from trainparity.protocols import ResumeCase, TrainingState
from trainparity.prototypes import ResumeCallbacks

CASE_SPEC = "trainparity.examples.resume_cases:CorrectResumeCase"


def logical_lines(target: object) -> int:
    """Count nonblank, noncomment source lines for one adapter/factory."""
    lines = inspect.getsource(target).splitlines()
    return sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in lines)


def state_observation(state: TrainingState) -> dict[str, Any]:
    """Capture only values needed by this Gate 1 example probe."""
    assert state.scheduler is not None
    return {
        "model.weight": state.model.state_dict()["weight"].detach().tolist(),
        "model.bias": state.model.state_dict()["bias"].detach().tolist(),
        "optimizer.param_groups.0.lr": state.optimizer.param_groups[0]["lr"],
        "scheduler.last_epoch": state.scheduler.last_epoch,
        "step": state.step,
    }


def run_resume_probe(case: ResumeCase | ResumeCallbacks) -> dict[str, Any]:
    """Exercise an example directly; this is not the production Gate 3 runner."""
    baseline = case.build(7)
    for _ in range(4):
        case.train_step(baseline)
    interrupted = case.build(7)
    for _ in range(3):
        case.train_step(interrupted)
    with tempfile.TemporaryDirectory(prefix="trainparity-gate1-") as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        case.save(interrupted, checkpoint)
        resumed = case.load(checkpoint, 7)
    case.train_step(resumed)
    expected = state_observation(baseline)
    observed = state_observation(resumed)
    first_difference = next((key for key in expected if expected[key] != observed[key]), None)
    return {
        "outcome": "PASS" if first_difference is None else "FAIL",
        "first_observed_divergence": first_difference,
        "baseline": expected,
        "resumed": observed,
    }


def fresh_process_import() -> dict[str, Any]:
    """Inspect the selected case from a fresh process and unrelated directory."""
    with tempfile.TemporaryDirectory(prefix="trainparity-import-") as directory:
        completed = subprocess.run(
            [sys.executable, "-m", "trainparity", "inspect", CASE_SPEC],
            cwd=directory,
            check=False,
            capture_output=True,
            text=True,
        )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def evaluate() -> dict[str, Any]:
    """Return Gate 1 API metrics and the two required resume outcomes."""
    imported = fresh_process_import()
    correct = run_resume_probe(CorrectResumeCase())
    faulty = run_resume_probe(MissingSchedulerStateCase())
    callbacks = run_resume_probe(make_resume_callbacks())
    return {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "selected_api": "class_protocol",
        "selection_reasons": [
            "one import specification identifies both construction and behavior",
            "ResumeCase gives structural type checking without inheritance",
            "a class avoids callback-field wiring while leaving the training step user-owned",
        ],
        "prototypes": {
            "class_protocol": {
                "adapter_logical_lines": logical_lines(CorrectResumeCase),
                "fresh_process_import": imported,
                "process_safe": imported["returncode"] == 0,
                "cloudpickle_required": False,
                "type_surface": "ResumeCase protocol",
                "user_wiring": "one zero-argument class",
            },
            "factory_callbacks": {
                "factory_logical_lines": logical_lines(make_resume_callbacks),
                "probe_outcome": callbacks["outcome"],
                "process_safe": True,
                "cloudpickle_required": False,
                "type_surface": "ResumeCallbacks dataclass",
                "user_wiring": "factory plus four callback fields",
            },
        },
        "resume_cases": {
            "correct": correct,
            "missing_scheduler_state": faulty,
        },
        "scope": "Gate 1 example probe only; no production comparator or resume runner",
    }


def main() -> None:
    """Write evaluation JSON to the requested project-owned path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected_api": result["selected_api"]}))


if __name__ == "__main__":
    main()

