"""Verify the Gate 4B production integration surface report and evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Gate 4B verification failed: {message}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(repo_root: Path, allow_pending_ci: bool = False) -> dict[str, Any]:
    json_path = repo_root / "artifacts/gate_reports/gate_4b.json"
    markdown_path = repo_root / "artifacts/gate_reports/gate_4b.md"
    report = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    _require(report["gate"] == "4B", "wrong gate")
    _require(report["outcome"] == "GO", "outcome is not GO")
    _require(not report["gate_5_started"], "Gate 5 was started")
    _require(len(report["projects"]) == 3, "expected three projects")
    totals = []
    required_timing = {
        "single_normal_run",
        "baseline_self_consistency",
        "candidate_save_exit_new_process_load_resume",
        "snapshot_capture",
        "serialization",
        "comparison",
        "total_wall",
        "end_to_end_multiplier",
    }
    for project in report["projects"]:
        loc = project["loc"]
        _require(loc["adapter_logical"] <= 30, f"{project['name']} adapter LOC")
        _require(loc["supporting_glue_logical"] <= 20, f"{project['name']} glue LOC")
        _require(loc["total_user_logical"] <= 50, f"{project['name']} total LOC")
        _require(loc["upstream_modified"] == 0, f"{project['name']} upstream modified")
        totals.append(loc["total_user_logical"])
        _require(project["fresh_clone"], f"{project['name']} was not a fresh clone")
        _require(project["clean"]["outcome"] == "PASS", f"{project['name']} clean")
        _require(project["fault"]["outcome"] == "FAIL", f"{project['name']} fault")
        _require(project["clean"]["fresh_resume_processes_distinct"], "fresh process")
        _require(required_timing <= set(project["clean"]["timing_seconds"]), "timings")
        _require(project["resources"]["wall_threshold_passed"], "wall threshold")
        _require(project["resources"]["artifact_threshold_passed"], "artifact threshold")
        _require(len(project["user_required_files"]) == 2, "hidden user-required file")
        _require(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in project["environment_propagated_keys"],
            "explicit TORCH_FORCE propagation",
        )
    _require(sorted(totals)[1] <= 40, "median total user LOC")
    baseline = report["baseline_comparison"]
    _require(baseline["functionally_closer_handwritten_logical_loc"] == 116, "baseline LOC")
    _require(baseline["trainparity_less_user_code_for_every_project"], "baseline comparison")
    _require(
        set(baseline["retained_semantics"])
        == {
            "model",
            "optimizer",
            "scheduler",
            "rng",
            "fresh_process",
            "first_observed_divergence",
            "PASS_FAIL_ABSTAIN_ERROR",
        },
        "retained semantics",
    )
    optimization = report["snapshot_optimization"]
    _require(not optimization["fingerprint_backend_introduced"], "fingerprint backend")
    _require(not optimization["comparison_semantics_changed"], "comparison weakened")
    _require(optimization["post_total_seconds"] < optimization["pre_total_seconds"], "profile")
    _require(not report["production_surface"]["framework_specific_branches"], "framework branch")
    faults = {fault["project"]: fault for fault in report["faults"]}
    _require(faults["nanogpt"]["classification"] == "trajectory-affecting", "nano fault")
    _require(faults["nanogpt"]["downstream_parameter_divergence_observed"], "nano params")
    for name in ("pytorch_examples_imagenet", "ignite_mnist_engine"):
        _require(faults[name]["classification"] == "control-state", f"{name} fault")
        _require(not faults[name]["downstream_parameter_divergence_observed"], f"{name} params")
    _require(report["test_summary"]["outcome"] == "PASS", "test summary")
    if not allow_pending_ci:
        _require(report["hosted_ci"]["conclusion"] == "success", "hosted CI")
    evidence = report["preservation"]["accepted_evidence_sha256"]
    for relative, expected in evidence.items():
        path = repo_root / relative
        _require(path.is_file() and _sha256(path) == expected, f"preservation: {relative}")
    _require(
        bool(re.fullmatch(r"[0-9a-f]{64}", report["preservation"]["user_document_sha256"])),
        "user document hash",
    )
    for phrase in (
        "Gate 5 was not started",
        "first observed divergences, not root-cause claims",
        "FullValueBackend remains",
    ):
        _require(phrase in markdown, f"Markdown phrase: {phrase}")
    return {"outcome": "PASS", "projects": 3, "median_total_user_loc": sorted(totals)[1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pending-ci", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(root, args.allow_pending_ci), sort_keys=True))


if __name__ == "__main__":
    main()
