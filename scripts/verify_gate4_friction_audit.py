"""Verify the dedicated Gate 4 friction rework report and preservation manifest."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Gate 4 friction audit verification failed: {message}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(repo_root: Path) -> dict[str, Any]:
    json_path = repo_root / "artifacts/gate_reports/gate_4_friction_audit.json"
    markdown_path = repo_root / "artifacts/gate_reports/gate_4_friction_audit.md"
    report = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    _require(report["gate"] == 4 and report["audit"] == "friction_rework", "wrong audit")
    _require(report["outcome"] == "PASS", "audit did not pass its evidence checks")
    _require(len(report["projects"]) == 3, "expected three projects")
    _require(report["metrics"]["fresh_clone_clean_passed"] == 3, "clean fresh clones")
    _require(report["metrics"]["upstream_modified_loc"] == 0, "upstream code changed")
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
        _require(project["fresh_clone"], f"{project['name']} was not a fresh clone")
        _require(project["clean_resume"]["outcome"] == "PASS", f"{project['name']} clean")
        _require(project["clean_resume"]["fresh_resume_processes_distinct"], "process boundary")
        _require(project["loc"]["upstream_modified"] == 0, "upstream LOC")
        _require(len(project["user_required_files"]) == 3, "hidden or missing user files")
        _require(
            project["loc"]["total_user_required"]
            == project["loc"]["user_required_adapter"]
            + project["loc"]["user_required_supporting_glue"],
            "user LOC arithmetic",
        )
        timings = project["clean_resume"]["timing_seconds"]
        _require(required_timing <= set(timings), f"{project['name']} timing phases")
        _require(all(timings[key] >= 0 for key in required_timing), "negative timing")
        _require(project["clean_resume"]["total_artifact_size_bytes"] > 0, "artifact size")
        _require(project["clean_resume"]["peak_rss_kib"] > 0, "peak RSS")
        _require(project["exact_commands"], "commands not recorded")
    categories = report["loc_inventory"]["per_project_categories"]
    _require(set(categories) == {project["name"] for project in report["projects"]}, "LOC split")
    _require(report["baselines"]["weak"]["logical_loc"] == 12, "weak baseline changed")
    closer = report["baselines"]["closer"]
    _require(closer["project_specific"], "closer baseline specificity")
    _require(closer["result"]["clean"]["outcome"] == "PASS", "closer clean")
    _require(closer["result"]["fault"]["outcome"] == "FAIL", "closer fault")
    _require(
        set(closer["result"]["fault"]["checked_components"])
        == {"model", "optimizer", "scheduler", "rng.torch_cpu"},
        "closer baseline state coverage",
    )
    faults = {item["project"]: item for item in report["faults"]}
    _require(faults["nanogpt"]["classification"] == "trajectory-affecting", "nano fault")
    _require(faults["nanogpt"]["downstream_parameter_divergence_observed"], "nano parameters")
    for name in ("pytorch_examples_imagenet", "ignite_mnist_engine"):
        _require(faults[name]["classification"] == "control-state", f"{name} fault")
        _require(not faults[name]["downstream_parameter_divergence_observed"], f"{name} params")
    _require(
        all(value is False for value in report["scope_guards"].values()),
        "scope guard violated",
    )
    evidence = report["preservation"]["accepted_evidence_sha256"]
    for relative, expected in evidence.items():
        path = repo_root / relative
        _require(path.is_file() and _sha256(path) == expected, f"preservation: {relative}")
    document_hash = report["preservation"]["user_document_sha256"]
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", document_hash)), "user document hash")
    for phrase in (
        "weak baseline",
        "not the total TrainParity overhead",
        "first observed divergences, not root-cause claims",
        "Gate 5 was not started",
    ):
        _require(phrase in markdown, f"Markdown phrase: {phrase}")
    return {
        "projects": len(report["projects"]),
        "fresh_clone_clean_passed": report["metrics"]["fresh_clone_clean_passed"],
        "accepted_evidence_files": len(evidence),
        "outcome": "PASS",
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(repo_root), sort_keys=True))


if __name__ == "__main__":
    main()
