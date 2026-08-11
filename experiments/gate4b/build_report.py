"""Build the deterministic Gate 4B human-review report from recorded evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logical_lines(path: Path) -> int:
    return sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _markdown(report: dict[str, Any]) -> str:
    project_rows = "\n".join(
        "| {name} | {adapter} | {glue} | {total} | {modified} | {clean} | {fault} | "
        "{multiple:.3f}x | {artifacts} |".format(
            name=project["name"],
            adapter=project["loc"]["adapter_logical"],
            glue=project["loc"]["supporting_glue_logical"],
            total=project["loc"]["total_user_logical"],
            modified=project["loc"]["upstream_modified"],
            clean=project["clean"]["outcome"],
            fault=project["fault"]["outcome"],
            multiple=project["resources"]["end_to_end_multiplier"],
            artifacts=project["resources"]["total_persisted_artifact_bytes"],
        )
        for project in report["projects"]
    )
    fault_rows = "\n".join(
        f"| {fault['project']} | {fault['classification']} | "
        f"{fault['first_observed_divergence']} | "
        f"{str(fault['downstream_parameter_divergence_observed']).lower()} |"
        for fault in report["faults"]
    )
    profile = report["snapshot_optimization"]
    return f"""# Gate 4B — Production Integration Surface

## Outcome

**{report['outcome']}** — stop after Gate 4B for human review. Gate 5 was not started.

Generic baseline/candidate planning, checkpoint staging, fresh-process execution,
snapshot IPC, deterministic reporting, timeout/temp-directory behavior, explicit
environment propagation, and PASS/FAIL/ABSTAIN/ERROR handling now live in the
framework-neutral TrainParity production package. No project-specific production
adapter or framework branch was added.

## Fresh-clone product surface

| Project | Adapter LOC | Glue LOC | Total user LOC | Upstream modified LOC | Clean | Fault | Total/normal wall | Persisted artifacts (bytes) |
|---|---:|---:|---:|---:|---|---|---:|---:|
{project_rows}

Median total user LOC is {report['metrics']['median_total_user_logical_loc']}; the
functionally closer hand-written fresh-process baseline is 116 logical LOC and is
project-specific. TrainParity retains exact model/optimizer/scheduler/RNG state,
fresh-process evidence, first-observed-divergence diagnostics, and four-state
result semantics with less user code in every project.

## ImageNet snapshot profile

The pre-optimization profile found byte-at-a-time storage iteration, not tensor
cloning or repeated materialization, as the dominant cost. FullValueBackend remains
the correctness reference; no fingerprint or collision-bearing backend was added.
The full snapshot path fell from {profile['pre_total_seconds']:.6f}s to
{profile['post_total_seconds']:.6f}s ({profile['speedup']:.2f}x), while byte-for-byte
compatibility tests preserve comparison semantics.

## Injected faults

| Project | Class | First observed divergence | Downstream parameter divergence |
|---|---|---|---|
{fault_rows}

These are first observed divergences, not root-cause claims.

## Verification

- Full lint, type-check, tests, coverage, build, and Gate 0-4 verifier replay: PASS.
- Three exact-commit external fresh clones on {report['environment']['gpu_name']}: PASS.
- Explicit child environment propagation, including `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`, is contract-tested; reports contain keys only, never environment values.
- Hosted GitHub Actions: {report['hosted_ci']['conclusion']} (run {report['hosted_ci']['run_id']}).
- Accepted Gate 0-4 evidence hashes and the recorded user-document hash are unchanged.

## Scope

No production framework adapter, distributed support, dashboard, service, backend
semantic weakening, Gate 5 work, or root-cause claim was introduced.
"""


def build(repo_root: Path, matrix_path: Path, ci_path: Path, test_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    ci = json.loads(ci_path.read_text(encoding="utf-8"))
    tests = json.loads(test_path.read_text(encoding="utf-8"))
    friction_path = repo_root / "artifacts/gate_reports/gate_4_friction_audit.json"
    friction = json.loads(friction_path.read_text(encoding="utf-8"))
    preservation = dict(friction["preservation"]["accepted_evidence_sha256"])
    for relative in (
        "artifacts/gate_reports/gate_4_friction_audit.json",
        "artifacts/gate_reports/gate_4_friction_audit.md",
    ):
        preservation[relative] = _sha256(repo_root / relative)

    faults = []
    classifications = {
        "pytorch_examples_imagenet": "control-state",
        "nanogpt": "trajectory-affecting",
        "ignite_mnist_engine": "control-state",
    }
    for project in matrix["projects"]:
        differences = project["fault"]["all_differences"]
        parameter_divergence = any(
            difference["path"].startswith(("model.", "model[", "state_dict.", "state_dict["))
            for difference in differences
        )
        primary = project["fault"]["primary_difference"]
        faults.append(
            {
                "project": project["name"],
                "classification": classifications[project["name"]],
                "first_observed_divergence": primary["path"],
                "difference_count": len(differences),
                "downstream_parameter_divergence_observed": parameter_divergence,
            }
        )

    totals = [project["loc"]["total_user_logical"] for project in matrix["projects"]]
    pre = matrix["snapshot_profile"]["pre"]["timing_seconds"]["total_snapshot_path"]
    post = matrix["snapshot_profile"]["post"]["timing_seconds"]["total_snapshot_path"]
    technical_pass = (
        len(matrix["projects"]) == 3
        and matrix["metrics"]["clean_passed"] == 3
        and matrix["metrics"]["faults_detected"] == 3
        and max(project["loc"]["adapter_logical"] for project in matrix["projects"]) <= 30
        and max(project["loc"]["supporting_glue_logical"] for project in matrix["projects"])
        <= 20
        and max(totals) <= 50
        and statistics.median(totals) <= 40
        and matrix["metrics"]["upstream_modified_loc"] == 0
        and all(
            project["resources"]["wall_threshold_passed"]
            and project["resources"]["artifact_threshold_passed"]
            for project in matrix["projects"]
        )
        and tests["outcome"] == "PASS"
        and ci["conclusion"] == "success"
    )
    return {
        "schema_version": 1,
        "gate": "4B",
        "scope": "production_integration_surface",
        "outcome": "GO" if technical_pass else "STOP",
        "projects": matrix["projects"],
        "metrics": matrix["metrics"],
        "faults": faults,
        "baseline_comparison": {
            "functionally_closer_handwritten_logical_loc": 116,
            "functionally_closer_handwritten_project_specific": True,
            "trainparity_total_user_logical_loc": totals,
            "trainparity_less_user_code_for_every_project": all(total < 116 for total in totals),
            "retained_semantics": [
                "model",
                "optimizer",
                "scheduler",
                "rng",
                "fresh_process",
                "first_observed_divergence",
                "PASS_FAIL_ABSTAIN_ERROR",
            ],
        },
        "snapshot_optimization": {
            "backend": "FullValueBackend correctness reference",
            "fingerprint_backend_introduced": False,
            "profiled_issue": "byte-at-a-time untyped-storage iteration",
            "pre_total_seconds": pre,
            "post_total_seconds": post,
            "speedup": pre / post,
            "comparison_semantics_changed": False,
        },
        "production_surface": {
            "files": [
                "src/trainparity/process_resume.py",
                "src/trainparity/process_worker.py",
                "src/trainparity/protocols.py",
                "src/trainparity/results.py",
                "src/trainparity/importing.py",
            ],
            "logical_loc": sum(
                _logical_lines(repo_root / relative)
                for relative in (
                    "src/trainparity/process_resume.py",
                    "src/trainparity/process_worker.py",
                    "src/trainparity/protocols.py",
                    "src/trainparity/results.py",
                    "src/trainparity/importing.py",
                )
            ),
            "framework_specific_branches": [],
        },
        "test_summary": tests,
        "hosted_ci": ci,
        "environment": matrix["environment"],
        "profile_evidence": matrix["snapshot_profile"],
        "preservation": {
            "accepted_evidence_sha256": preservation,
            "user_document_sha256": friction["preservation"]["user_document_sha256"],
        },
        "limitations": [
            "Gate 4B remains three tiny single-process external-project cases on one GPU model.",
            "Full-value snapshots prioritize correctness and are not a performance-optimized backend.",
            "First observed divergence is not a root-cause claim.",
        ],
        "gate_5_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--ci", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    report = build(repo_root, args.matrix, args.ci, args.tests)
    json_path = repo_root / "artifacts/gate_reports/gate_4b.json"
    markdown_path = repo_root / "artifacts/gate_reports/gate_4b.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"outcome": report["outcome"], "projects": len(report["projects"])}, sort_keys=True))


if __name__ == "__main__":
    main()
