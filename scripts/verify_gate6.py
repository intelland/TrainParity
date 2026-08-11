"""Dedicated Gate 6 inclusion-decision verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED = {
    "world_size_1",
    "world_size_2",
    "world_size_3",
    "world_size_4",
    "non_divisible",
    "drop_last",
    "padding_duplicate",
    "missing_ids",
    "same_rank_duplication",
    "cross_rank_overlap",
    "custom_sampler",
    "finite_iterable_dataset",
    "unknown_universe_abstain",
    "multi_epoch_shuffle",
}


def _require(value: bool, message: str) -> None:
    if not value:
        raise SystemExit(f"Gate 6 verification failed: {message}")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path, allow_pending_ci: bool = False) -> dict[str, Any]:
    """Verify the frozen Gate 6 report and all recorded acceptance evidence."""
    report = _load(root / "artifacts" / "gate_reports" / "gate_6.json")
    markdown = (root / "artifacts" / "gate_reports" / "gate_6.md").read_text(
        encoding="utf-8"
    )
    matrix = _load(root / "experiments" / "gate6" / "recorded" / "cpu_matrix.json")
    product = _load(
        root / "experiments" / "gate6" / "recorded" / "product_surface.json"
    )
    _require(
        report["status"] == "PASS" and report["module_decision"] == "INCLUDE_MODULE",
        "module decision",
    )
    _require(not report["gate_7_started"], "Gate 7 scope")
    _require(
        set(report["policies"])
        == {
            "exactly_once",
            "at_least_once",
            "no_cross_rank_overlap",
            "expected_padding",
        },
        "policy inventory",
    )
    _require(
        matrix["device"] == "cpu" and not report["metrics"]["gpu_work_added"],
        "CPU-only scope",
    )
    _require(len(matrix["rows"]) == 17, "matrix row count")
    _require(
        all(row["result"]["outcome"] == row["expected"] for row in matrix["rows"]),
        "matrix outcomes",
    )
    exercised = {item for row in matrix["rows"] for item in row["requirements"]}
    _require(exercised >= REQUIRED, f"required fixture coverage: {sorted(REQUIRED - exercised)}")
    rows = {row["name"]: row["result"] for row in matrix["rows"]}
    _require(rows["unknown_universe"]["outcome"] == "ABSTAIN", "unknown universe")
    _require(
        rows["same_rank_duplicate"]["same_rank_duplicate_id_count"] == 1
        and rows["same_rank_duplicate"]["cross_rank_overlap_id_count"] == 0,
        "same-rank distinction",
    )
    _require(
        rows["cross_rank_overlap"]["cross_rank_overlap_id_count"] == 1
        and rows["cross_rank_overlap"]["same_rank_duplicate_id_count"] == 0,
        "cross-rank distinction",
    )
    _require(
        rows["non_divisible_padding"]["actual_padding_count"]
        == rows["non_divisible_padding"]["expected_padding_count"]
        == 2,
        "padding counts",
    )
    padding = _load(
        root
        / "experiments"
        / "gate6"
        / "recorded"
        / "machine_evidence"
        / "non_divisible_padding.json"
    )
    _require(
        len(padding["repeated_ids"]) == 2
        and all(item["ranks"] for item in padding["repeated_ids"]),
        "padding IDs/ranks",
    )
    _require(bool(padding["traces"]), "complete trajectories")
    _require(len(product["cases"]) == 2, "product cases")
    for case in product["cases"]:
        _require(case["result"]["outcome"] == "PASS", f"{case['case']} outcome")
        _require(case["user_required_logical_loc"] <= 25, f"{case['case']} LOC")
        _require(case["upstream_modified_loc"] == 0, f"{case['case']} upstream")
    baseline = report["counter_baseline"]
    _require(
        baseline["logical_loc"] == 11 and len(baseline["structural_benefits"]) >= 3,
        "Counter comparison",
    )
    semantics = report["semantics"]
    _require(
        semantics["terminal_output_bounded"] and semantics["complete_evidence_optional"],
        "bounded output",
    )
    _require(semantics["first_observed_not_root_cause"], "first-observed language")
    carry = report["gate5_carry_forward"]
    _require(
        carry["accepted_gate_0_through_4b_hashes_verified_unchanged"]
        and not carry["gpu_rerun_for_carry_forward"],
        "Gate 5 carry-forward",
    )
    tied = carry["tied_parameter_observation_scope"]
    _require(
        len(tied["optimizer_excludes"]) == 2
        and len(tied["full_model_parameters_include"]) == 2
        and len(tied["full_gradient_includes"]) == 2,
        "tied observation scope",
    )
    _require(
        not tied["production_project_specific_mapping_rule_added"],
        "project-specific optimizer rule",
    )
    production = (root / "src" / "trainparity" / "sample_coverage.py").read_text(
        encoding="utf-8"
    ).lower()
    for forbidden in (
        "nanogpt",
        "imagenet",
        "ignite",
        "nccl",
        "slurm",
        "init_process_group",
    ):
        _require(forbidden not in production, f"production branch {forbidden}")
    for relative, expected in report["preservation"]["accepted_evidence_sha256"].items():
        path = root / relative
        _require(path.is_file() and _hash(path) == expected, f"preservation {relative}")
    document_hash = _hash(root / "CODEX_REMOTE_DEVELOPMENT.md")
    allowed = {
        report["preservation"]["tracked_remote_development_sha256"],
        report["preservation"]["user_uncommitted_remote_development_sha256"],
    }
    _require(document_hash in allowed, "remote development document")
    if not allow_pending_ci:
        _require(report["test_summary"]["outcome"] == "PASS", "test summary")
        _require(report["hosted_ci"]["conclusion"] == "success", "hosted CI")
    for phrase in (
        "INCLUDE_MODULE",
        "Unknown-universe",
        "First observed violations are not root-cause claims",
        "11-line Counter",
    ):
        _require(phrase in markdown, f"Markdown phrase {phrase}")
    return {
        "status": "PASS",
        "recommendation": "INCLUDE_MODULE",
        "rows": 17,
        "product_cases": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pending-ci", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(root, arguments.allow_pending_ci), sort_keys=True))


if __name__ == "__main__":
    main()
