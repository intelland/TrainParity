"""Run same-allocation Gate 3 CUDA RNG and GradScaler checks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from trainparity.runner import ResumeRunner

PREFIX = "trainparity.examples.gate3_cases:"
CASES = {
    "clean": ("DeterministicCase", "PASS", None),
    "missing_cuda_rng": ("MissingCudaRngCase", "FAIL", "rng.torch_cuda"),
    "missing_grad_scaler": ("MissingGradScalerCase", "FAIL", "scaler"),
}


def run_matrix(output: Path, *, repeats: int = 3) -> dict[str, Any]:
    """Execute all cases on the one CUDA device visible in this Slurm job."""
    if not torch.cuda.is_available():
        raise RuntimeError("a real CUDA device is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Gate 3 GPU matrix requires exactly one visible CUDA device")
    os.environ["TRAINPARITY_DEVICE"] = "cuda"
    output.parent.mkdir(parents=True, exist_ok=True)
    run_root = output.parent / "gpu_runs"
    records: list[dict[str, Any]] = []
    runner = ResumeRunner(timeout=120.0)
    for name, (case, expected_outcome, expected_component) in CASES.items():
        runs = [
            runner.run(PREFIX + case, work_dir=run_root / f"{name}_{index}").to_dict()
            for index in range(repeats)
        ]
        signatures = {
            (
                result["outcome"],
                result["first_divergent_step"],
                (result.get("primary_difference") or {}).get("path"),
            )
            for result in runs
        }
        matched = all(result["outcome"] == expected_outcome for result in runs)
        component_matched = True
        if expected_component is not None:
            component_matched = all(
                (result.get("primary_difference") or {}).get("path", "").startswith(
                    expected_component
                )
                for result in runs
            )
            matched = matched and component_matched
        records.append(
            {
                "name": name,
                "case": case,
                "expected_outcome": expected_outcome,
                "expected_component": expected_component,
                "matched": matched,
                "component_matched": component_matched,
                "stable": len(signatures) == 1,
                "runs": runs,
            }
        )
    device = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": 1,
        "gate": 3,
        "device": "cuda",
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": device.name,
            "gpu_total_memory": device.total_memory,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": platform.node(),
        },
        "same_device_policy": "all A/B workers inherit one CUDA_VISIBLE_DEVICES value",
        "repeat_count": repeats,
        "cases": records,
        "all_matched": all(record["matched"] and record["stable"] for record in records),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = run_matrix(args.output, repeats=args.repeats)
    print(json.dumps({"all_matched": report["all_matched"], "environment": report["environment"]}, sort_keys=True))
    if not report["all_matched"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
