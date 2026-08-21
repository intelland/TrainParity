"""Dedicated Gate 5 evidence verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FAULTS = {
    "missing_accumulation_scaling",
    "variable_length_mean_of_means",
    "optimizer_step_per_microbatch",
    "scheduler_step_per_microbatch",
    "zero_grad_wrong_time",
    "gradient_clip_wrong_time",
    "amp_unscale_scaler_timing",
    "incomplete_final_window",
}


def _require(value: bool, message: str) -> None:
    if not value:
        raise SystemExit(f"Gate 5 verification failed: {message}")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify(root: Path, allow_pending_ci: bool = False) -> dict[str, Any]:
    report = json.loads((root / "artifacts/gate_reports/gate_5.json").read_text(encoding="utf-8"))
    markdown = (root / "artifacts/gate_reports/gate_5.md").read_text(encoding="utf-8")
    _require(report["outcome"] == "GO" and not report["gate_6_started"], "scope/outcome")
    contract = report["contract"]
    _require(contract["equivalence_is_user_declared"], "equivalence declaration")
    _require(contract["first_observed_not_root_cause"], "root-cause language")
    _require(not contract["tolerance_inferred_or_tuned"], "inferred tolerance")
    _require(contract["loss_accounting_optional"], "optional loss accounting")
    _require(len(contract["observed_phases"]) == 5, "bounded phase inventory")
    metrics = report["metrics"]
    _require(metrics["clean_false_positives"] == 0, "clean false positive")
    _require(metrics["faults_detected"] == metrics["fault_count"] == 8, "eight faults")
    _require(metrics["cpu_repeats"] == 3 and metrics["cpu_rows"] == 30, "CPU repeats")
    _require(metrics["all_fresh_process_ids_distinct"], "fresh processes")
    _require(metrics["all_initial_states_verified_equal"], "initial state")
    _require(metrics["peak_temporary_directory_bytes"] > 0, "peak temporary disk")
    _require(metrics["recorded_persisted_artifact_bytes"] > 0, "persisted artifacts")
    _require({row["name"] for row in report["faults"]} == FAULTS, "fault inventory")
    _require(all(row["detected"] for row in report["faults"]), "fault detection")
    _require(report["gpu"]["same_device_only"] and not report["gpu"]["cross_gpu_model_comparison"], "GPU semantics")
    _require(report["gpu"]["name"] == "NVIDIA L40S" and report["gpu"]["slurm_job_id"] == "58980407", "GPU evidence")
    projects = report["product_surface"]["projects"]
    _require(len(projects) == 2, "product project count")
    for project in projects:
        _require(project["total_user_logical_loc"] <= 50, f"{project['project']} LOC")
        _require(project["upstream_modified_loc"] == 0, f"{project['project']} upstream")
        _require(project["result"]["outcome"] == "PASS", f"{project['project']} clean")
    _require(report["known_non_equivalence"]["batchnorm_training"]["outcome"] == "FAIL", "BatchNorm control")
    _require(report["strict_control"]["nanogpt_tied_parameter_mapping"] == "ABSTAIN", "ambiguous optimizer control")
    summary = report["test_summary"]
    _require(allow_pending_ci or summary["outcome"] == "PASS", "test summary")
    _require(allow_pending_ci or report["hosted_ci"]["conclusion"] == "success", "hosted CI")
    for relative, expected in report["preservation"]["accepted_evidence_sha256"].items():
        path = root / relative
        _require(path.is_file() and _hash(path) == expected, f"preservation {relative}")
    preservation = report["preservation"]
    for key in (
        "tracked_remote_development_sha256",
        "user_uncommitted_remote_development_sha256",
    ):
        _require(_is_sha256(preservation.get(key)), f"historical document hash {key}")
    for phrase in ("user-declared", "first observed divergences, not root-cause claims", "FullValueBackend remains"):
        _require(phrase in markdown, f"Markdown phrase {phrase}")
    return {"outcome": "PASS", "faults": 8, "projects": 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pending-ci", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(root, arguments.allow_pending_ci), sort_keys=True))


if __name__ == "__main__":
    main()
