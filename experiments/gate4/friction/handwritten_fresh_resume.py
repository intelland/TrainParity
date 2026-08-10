"""Project-specific hand-written check over Ignite fresh-process resume outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def _first_difference(left: object, right: object, path: str) -> str | None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            return path
        return None
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right), key=str)
        for key in keys:
            if key not in left or key not in right:
                return f"{path}.{key}"
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
    ):
        if len(left) != len(right):
            return path
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if type(left) is type(right) and left == right else path


def _state(checkpoint: Path) -> dict[str, object]:
    value: dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "model": value["model"],
        "optimizer": value["optimizer"],
        "scheduler": value["lr_scheduler"],
        "rng": {"torch_cpu": torch.load(checkpoint.with_name("rng_state.pt"), weights_only=True)},
    }


def _compare(left: Path, right: Path) -> dict[str, object]:
    left_state, right_state = _state(left), _state(right)
    first = _first_difference(left_state, right_state, "$")
    model_difference = _first_difference(left_state["model"], right_state["model"], "$.model")
    return {
        "outcome": "PASS" if first is None else "FAIL",
        "first_observed_divergence": first,
        "downstream_parameter_divergence": model_difference is not None,
        "checked_components": ["model", "optimizer", "scheduler", "rng.torch_cpu"],
    }


def run(checkout: Path, workspace: Path, user_files: Path, output: Path) -> dict[str, object]:
    sys.path.insert(0, str(user_files))
    import trainparity_adapter as adapter
    import trainparity_project_glue as project

    baseline = adapter.checkpoint_path(workspace / "normal_a")
    clean = adapter.checkpoint_path(workspace / "candidate_resume")
    clean_result = _compare(baseline, clean)

    fault_dir = workspace / "handwritten_fault_resume"
    fault_dir.mkdir(parents=True, exist_ok=True)
    fault_input = adapter.checkpoint_path(fault_dir)
    shutil.copy2(adapter.checkpoint_path(workspace / "candidate_split"), fault_input)
    checkpoint: dict[str, Any] = torch.load(fault_input, map_location="cpu", weights_only=False)
    checkpoint["lr_scheduler"]["last_epoch"] -= 1
    torch.save(checkpoint, fault_input)
    command = project.command(checkout, fault_dir, project.TOTAL_STEP, fault_input)
    stdout_path, stderr_path = fault_dir / "stdout.log", fault_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=checkout,
            env=os.environ,
            stdout=stdout,
            stderr=stderr,
            timeout=300,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:])
    fault_result = _compare(baseline, adapter.checkpoint_path(fault_dir))
    result: dict[str, object] = {
        "project": "ignite_mnist_engine",
        "project_specific": True,
        "fresh_process_resume": True,
        "resume_process_pid": int((fault_dir / "process.pid").read_text(encoding="utf-8")),
        "clean": clean_result,
        "fault": fault_result,
        "diagnostic": "equivalent selected state"
        if fault_result["outcome"] == "PASS"
        else f"first observed divergence at {fault_result['first_observed_divergence']}",
        "command": command,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--user-files", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.checkout, args.workspace, args.user_files, args.output)
    print(json.dumps({"clean": result["clean"], "fault": result["fault"]}, sort_keys=True))
    if result["clean"]["outcome"] != "PASS" or result["fault"]["outcome"] != "FAIL":  # type: ignore[index]
        raise SystemExit(1)


if __name__ == "__main__":
    main()
