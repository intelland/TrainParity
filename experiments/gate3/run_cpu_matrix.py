"""Run the formal Gate 3 CPU fault and control matrix."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from trainparity.runner import ResumeRunner

PREFIX = "trainparity.examples.gate3_cases:"
FAULTS = {
    "missing_model": ("MissingModelCase", 2, "model"),
    "missing_optimizer": ("MissingOptimizerCase", 2, "optimizer"),
    "missing_scheduler": ("MissingSchedulerCase", 2, "scheduler"),
    "missing_python_rng": ("MissingPythonRngCase", 2, "rng.python"),
    "missing_numpy_rng": ("MissingNumpyRngCase", 2, "rng.numpy"),
    "missing_torch_cpu_rng": ("MissingTorchCpuRngCase", 2, "rng.torch_cpu"),
    "data_cursor_offset": ("CursorOffsetCase", 3, "batch.sample_ids"),
    "resume_step_off_by_one": ("StepOffByOneCase", 2, "step"),
    "optimizer_parameter_group_mismatch": ("OptimizerGroupMismatchCase", 2, "optimizer"),
    "extra_scheduler_step": ("ExtraSchedulerStepCase", 2, "scheduler"),
    "missing_hidden_module_global": ("MissingHiddenGlobalCase", 2, "extra.hidden_module_counter"),
}


def _signature(result: dict[str, Any]) -> tuple[object, ...]:
    primary = result.get("primary_difference") or {}
    return (
        result.get("outcome"),
        result.get("first_divergent_step"),
        primary.get("path"),
        primary.get("reason"),
    )


def run_matrix(output: Path, *, repeats: int = 3) -> dict[str, Any]:
    """Execute controls and formal faults and write complete JSON evidence."""
    if repeats < 3:
        raise ValueError("Gate 3 formal evidence requires at least three repeats")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_root = output.parent / "cpu_runs"
    runner = ResumeRunner(timeout=90.0)
    clean_runs = [
        runner.run(PREFIX + "DeterministicCase", work_dir=run_root / f"clean_{index}").to_dict()
        for index in range(repeats)
    ]
    fault_records: list[dict[str, Any]] = []
    for name, (case, expected_step, expected_component) in FAULTS.items():
        runs = [
            runner.run(PREFIX + case, work_dir=run_root / f"{name}_{index}").to_dict()
            for index in range(repeats)
        ]
        signatures = {_signature(result) for result in runs}
        observed = runs[0].get("primary_difference") or {}
        fault_records.append(
            {
                "name": name,
                "case": case,
                "expected_step": expected_step,
                "expected_component": expected_component,
                "detected": all(result["outcome"] == "FAIL" for result in runs),
                "component_matched": all(
                    result.get("first_divergent_step") == expected_step
                    and (result.get("primary_difference") or {}).get("path", "").startswith(
                        expected_component
                    )
                    for result in runs
                ),
                "stable": len(signatures) == 1,
                "observed_step": runs[0].get("first_divergent_step"),
                "observed_component": observed.get("path"),
                "runs": runs,
            }
        )
    abstain = runner.run(
        PREFIX + "NondeterministicCase", work_dir=run_root / "nondeterministic"
    ).to_dict()
    error = runner.run(
        PREFIX + "ChildExceptionCase", work_dir=run_root / "child_exception"
    ).to_dict()
    report = {
        "schema_version": 1,
        "gate": 3,
        "device": "cpu",
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "step_semantics": "snapshot step N is after exactly N completed optimizer updates",
        "phase": "completed_training_step",
        "total_steps_per_trajectory": 4,
        "split_step": 2,
        "repeat_count": repeats,
        "clean": clean_runs,
        "faults": fault_records,
        "nondeterministic_control": abstain,
        "child_exception_control": error,
        "metrics": {
            "clean_false_positives": sum(result["outcome"] != "PASS" for result in clean_runs),
            "fault_count": len(fault_records),
            "faults_detected": sum(record["detected"] for record in fault_records),
            "components_matched": sum(record["component_matched"] for record in fault_records),
            "stable_faults": sum(record["stable"] for record in fault_records),
            "distinct_resume_pids": all(
                result["pre_save"]["pid"] != result["post_load"]["pid"]
                for result in clean_runs
            ),
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = run_matrix(args.output, repeats=args.repeats)
    print(json.dumps(report["metrics"], sort_keys=True))
    if (
        report["metrics"]["clean_false_positives"]
        or report["metrics"]["faults_detected"] != len(FAULTS)
        or report["metrics"]["stable_faults"] != len(FAULTS)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

