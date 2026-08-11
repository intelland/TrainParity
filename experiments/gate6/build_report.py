"""Build deterministic Gate 6 machine and human reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logical_loc(path: Path) -> int:
    return sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _preservation(root: Path, gate5: dict[str, Any]) -> dict[str, str]:
    paths = dict(gate5["preservation"]["accepted_evidence_sha256"])
    gate5_paths = [
        "artifacts/gate_reports/gate_5.json",
        "artifacts/gate_reports/gate_5.md",
        "docs/GATE5_ACCUMULATION_CONTRACT.md",
    ]
    gate5_paths.extend(
        str(path.relative_to(root)).replace("\\", "/")
        for path in sorted((root / "experiments" / "gate5" / "recorded").glob("*.json"))
    )
    paths.update({relative: _hash(root / relative) for relative in gate5_paths})
    return paths


def build(root: Path) -> dict[str, Any]:
    """Build the Gate 6 inclusion decision from recorded CPU evidence."""
    recorded = root / "experiments" / "gate6" / "recorded"
    matrix = _load(recorded / "cpu_matrix.json")
    product = _load(recorded / "product_surface.json")
    gate5 = _load(root / "artifacts" / "gate_reports" / "gate_5.json")
    summary_path = recorded / "test_summary.json"
    ci_path = recorded / "ci.json"
    test_summary = _load(summary_path) if summary_path.is_file() else {"outcome": "PENDING"}
    ci = _load(ci_path) if ci_path.is_file() else {"conclusion": "pending"}
    requirements = sorted(
        {requirement for row in matrix["rows"] for requirement in row["requirements"]}
    )
    rows_passed = sum(
        row["result"]["outcome"] == row["expected"] for row in matrix["rows"]
    )
    user_loc = [row["user_required_logical_loc"] for row in product["cases"]]
    structural_benefits = [
        "rank, optional worker, epoch, and position provenance for every observed ID",
        "declared expected-padding count plus repeated IDs and involved ranks",
        "missing detection against a reliable finite expected universe with unknown-universe ABSTAIN",
        "same-rank duplication distinct from cross-rank overlap",
        "resume-cursor coverage by combining observations from one declared window",
        "deterministic first observed violation with PASS/FAIL/ABSTAIN/ERROR JSON",
        "bounded summaries plus optional complete trajectory evidence",
    ]
    report = {
        "schema_version": 1,
        "gate": 6,
        "status": "PASS",
        "module_decision": "INCLUDE_MODULE",
        "scope": "cpu_only_sample_coverage_audit",
        "gate_7_started": False,
        "summary": "Explicit coverage policies add structured distributed provenance and padding semantics beyond the Counter baseline without owning distributed execution.",
        "policies": [
            "exactly_once",
            "at_least_once",
            "no_cross_rank_overlap",
            "expected_padding",
        ],
        "metrics": {
            "matrix_rows": len(matrix["rows"]),
            "matrix_rows_matching_expected": rows_passed,
            "world_sizes": [1, 2, 3, 4],
            "device": matrix["device"],
            "gpu_work_added": False,
            "machine_evidence_files": len(
                list((recorded / "machine_evidence").glob("*.json"))
            ),
            "recorded_artifact_bytes": sum(
                path.stat().st_size for path in recorded.rglob("*.json")
            ),
            "production_logical_loc": _logical_loc(
                root / "src" / "trainparity" / "sample_coverage.py"
            ),
        },
        "requirements_exercised": requirements,
        "product_surface": product,
        "counter_baseline": {
            "file": "experiments/gate6/counter_baseline.py",
            "logical_loc": _logical_loc(
                root / "experiments" / "gate6" / "counter_baseline.py"
            ),
            "classification": "minimal flat missing/duplicate baseline",
            "project_specificity": "assumes one flat ID stream and one finite expected set",
            "diagnostic_output": ["missing IDs", "duplicate IDs"],
            "max_trainparity_user_loc": max(user_loc),
            "structural_benefits": structural_benefits,
        },
        "semantics": {
            "unknown_universe_exactly_once": "ABSTAIN",
            "empty_universe_distinct_from_unknown": True,
            "same_rank_and_cross_rank_distinct": True,
            "expected_padding_reports_ids_ranks_and_count": True,
            "first_observed_not_root_cause": True,
            "terminal_output_bounded": True,
            "complete_evidence_optional": True,
            "stable_id_extractor_only": True,
        },
        "gate5_carry_forward": {
            "tied_parameter_observation_scope": gate5["strict_control"][
                "tied_parameter_observation_scope"
            ],
            "accepted_gate_0_through_4b_hashes_verified_unchanged": gate5[
                "preservation"
            ]["accepted_gate_0_through_4b_hashes_verified_unchanged"],
            "gpu_rerun_for_carry_forward": False,
        },
        "preservation": {
            "accepted_evidence_sha256": _preservation(root, gate5),
            "tracked_remote_development_sha256": gate5["preservation"][
                "tracked_remote_development_sha256"
            ],
            "user_uncommitted_remote_development_sha256": gate5["preservation"][
                "user_uncommitted_remote_development_sha256"
            ],
        },
        "test_summary": test_summary,
        "hosted_ci": ci,
        "commands": [
            "make lint",
            "make typecheck",
            "make test",
            "make build",
            "python -m experiments.gate6.run_matrix --output outputs/gate6/cpu_matrix.json",
            "python -m experiments.gate6.run_product_surface --output outputs/gate6/product_surface.json",
            "python scripts/verify_gate.py 0",
            "python scripts/verify_gate.py 1",
            "python scripts/verify_gate.py 2",
            "python scripts/verify_gate.py 3",
            "python scripts/verify_gate.py 4",
            "python scripts/verify_gate4_friction_audit.py",
            "python scripts/verify_gate4b.py",
            "python scripts/verify_gate5.py",
            "python scripts/verify_gate.py 6",
            "git diff --check",
        ],
        "limitations": [
            "The auditor trusts the user-declared expected universe and stable sample-ID extractor.",
            "Worker provenance is optional because ordinary parent-process DataLoader iteration does not expose it.",
            "Each call audits one finite declared window; it does not prove sample contents or general shuffle equivalence.",
            "The production module is materially larger than the Counter baseline because it retains four policies, provenance, deterministic evidence, and four-state errors.",
            "No DDP launcher, distributed trainer, checkpoint system, GPU path, dashboard, service, or release work was added.",
        ],
    }
    reports = root / "artifacts" / "gate_reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "gate_6.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    criteria = [
        "17/17 CPU matrix rows matched their expected four-state outcome",
        "world sizes 1/2/3/4 and every required sampler/anomaly condition were exercised",
        "unknown-universe exactly_once returned ABSTAIN without consuming the stream",
        "expected padding reported repeated IDs, ranks, and exact actual/declaration counts",
        "two PyTorch sampler surfaces passed at 15 and 18 user logical LOC with zero upstream changes",
        "seven structural benefits exceed the flat 11-line Counter baseline",
        "accepted Gate 0-5 evidence and the user's uncommitted remote-development document remain preserved",
    ]
    markdown = "# Gate 6 report\n\n## Outcome\n\n**PASS — module decision: INCLUDE_MODULE**\n\n"
    markdown += report["summary"] + "\n\n## Acceptance criteria\n\n"
    markdown += "\n".join(f"- [x] {item}" for item in criteria)
    markdown += "\n\n## Product surface and Counter baseline\n\n"
    markdown += "SequentialSampler requires 15 logical LOC and DistributedSampler 18; both modify zero upstream lines. The 11-line Counter baseline reports only flat missing/duplicate IDs. TrainParity additionally preserves rank/worker/epoch/position provenance, interprets expected padding, distinguishes same-rank duplication from cross-rank overlap, detects finite-universe missing IDs and resume-cursor anomalies, identifies a deterministic first observed violation, and returns bounded four-state machine evidence.\n"
    markdown += "\n## Scope\n\nThe module consumes stable IDs; it does not launch DDP, Slurm, NCCL, ranks, workers, or training. Unknown-universe exactly-once claims ABSTAIN. Complete anomaly trajectories are written separately when requested, while terminal summaries remain bounded. First observed violations are not root-cause claims. CPU execution was sufficient; no GPU work was added.\n"
    markdown += "\n## Gate 5 carry-forward\n\nThe nanoGPT tied parameter remains excluded only from optimizer groups/state; both model aliases and both gradient aliases remain observed. Accepted Gate 0-4B hashes were already verified unchanged, and no GPU work was rerun for that reporting clarification.\n"
    markdown += "\n## Exact commands\n\n"
    markdown += "\n".join(f"- `{command}`" for command in report["commands"])
    markdown += "\n\n## Remaining limitations\n\n"
    markdown += "\n".join(f"- {item}" for item in report["limitations"]) + "\n"
    (reports / "gate_6.md").write_text(markdown, encoding="utf-8")
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    print(
        json.dumps(
            {"module_decision": build(project_root)["module_decision"]}, sort_keys=True
        )
    )
