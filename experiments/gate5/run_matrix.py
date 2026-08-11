"""Run the formal Gate 5 CPU or same-device GPU matrix."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from trainparity import ToleranceComparison
from trainparity.accumulation import (
    AccumulationExecutionPlan,
    AccumulationRunner,
)


def _cases(device: str) -> list[tuple[str, str, AccumulationExecutionPlan, str]]:
    clean = AccumulationExecutionPlan(2)
    cases = [
        ("clean_linear", "experiments.gate5.cases:LinearCase", clean, "PASS"),
        ("clean_mlp", "experiments.gate5.cases:MLPCase", clean, "PASS"),
        ("clean_token", "experiments.gate5.cases:TokenCase", clean, "PASS"),
        ("missing_accumulation_scaling", "experiments.gate5.cases:LinearCase", AccumulationExecutionPlan(2, scale_accumulated_loss=False, use_explicit_loss_accounting=False), "FAIL"),
        ("variable_length_mean_of_means", "experiments.gate5.cases:TokenCase", AccumulationExecutionPlan(2, use_explicit_loss_accounting=False), "FAIL"),
        ("optimizer_step_per_microbatch", "experiments.gate5.cases:LinearCase", AccumulationExecutionPlan(2, optimizer_step_per_microbatch=True), "FAIL"),
        ("scheduler_step_per_microbatch", "experiments.gate5.cases:LinearCase", AccumulationExecutionPlan(2, scheduler_step_per_microbatch=True), "FAIL"),
        ("zero_grad_wrong_time", "experiments.gate5.cases:LinearCase", AccumulationExecutionPlan(2, zero_grad_before_gradient_observation=True), "FAIL"),
        ("gradient_clip_wrong_time", "experiments.gate5.cases:ClipCase", AccumulationExecutionPlan(2, clip_grad_norm=1.0, clip_per_microbatch=True), "FAIL"),
        ("incomplete_final_window", "experiments.gate5.cases:LinearCase", AccumulationExecutionPlan(2, omit_final_microbatch=True), "FAIL"),
    ]
    if device.startswith("cuda"):
        cases.extend([
            ("clean_amp", "experiments.gate5.cases:AmpCase", clean, "PASS"),
            ("amp_unscale_scaler_timing", "experiments.gate5.cases:AmpCase", AccumulationExecutionPlan(2, amp_step_before_unscale=True), "FAIL"),
        ])
    return cases


def run(output: Path, device: str, repeats: int) -> dict[str, Any]:
    """Execute fresh-process cases with a fixed, user-selected policy."""
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = output.parent / (output.stem + "_runs")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for repeat in range(repeats):
        for name, case, plan, expected in _cases(device):
            policy = ToleranceComparison(rtol=1e-6, atol=1e-7)
            report = raw_dir / f"{name}_{repeat}.json"
            result = AccumulationRunner(comparison=policy).run(
                case, candidate=plan, device=device, report_path=report
            )
            rows.append({
                "name": name,
                "repeat": repeat,
                "expected": expected,
                "result": result.to_dict(),
            })
    payload = {
        "schema_version": 1,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "repeats": repeats,
        "duration_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    payload = run(arguments.output, arguments.device, arguments.repeats)
    mismatches = [row for row in payload["rows"] if row["result"]["outcome"] != row["expected"]]
    print(json.dumps({"rows": len(payload["rows"]), "mismatches": len(mismatches)}))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
