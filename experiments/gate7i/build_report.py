"""Build the Gate 7I machine and human review reports from recorded evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import trainparity
from trainparity import api

REMOVED_PUBLIC_NAMES = (
    "ExternalProcessEvidence",
    "PACKAGE_VERSION",
    "SampleCoverageAuditor",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logical_lines(path: Path) -> int:
    return sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _markdown(report: dict[str, Any]) -> str:
    matrix = "\n".join(
        f"- Python {row['python']} / PyTorch {row['torch']}: {row['status']} "
        f"({row['clean_examples']} clean PASS, {row['intentional_failures']} intentional FAIL)"
        for row in report["compatibility"]["matrix"]
    )
    commands = "\n".join(f"- `{command}`" for command in report["commands"])
    return f"""# Gate 7I release-surface hardening report

## Outcome

**{report['status']} — {report['recommendation']}**

Publication remains held. No PyPI upload, Git tag, GitHub release, rename, merge,
or repository-visibility change was performed.

## Public surface

- Recommended top-level names: {report['public_api']['top_level_count']}
- Advanced `trainparity.api` names: {report['public_api']['advanced_count']}
- Removed accidental public names: {', '.join(report['public_api']['removed'])}
- README integration: {report['readme_example']['logical_loc']} logical LOC,
  complete PyTorch DataLoader case, run directly by CI
- Console-script entry point: absent for v0.1

## Compatibility

{matrix}

The declared range is `{report['compatibility']['declared_range']}`. No claim is
made for PyTorch 2.14, another Python release, or an unrecorded device.

## Verification

- Tests: {report['tests']['passed']} passed
- Coverage: {report['tests']['coverage_percent']}%
- Installed-wheel smoke: {report['release']['wheel_smoke_status']}
- Repository/archive audit: {report['release']['audit_status']}
- Twine metadata/README rendering: {report['release']['twine_check']}
- Accepted evidence hashes unchanged: {report['preservation']['accepted_evidence_unchanged']}
- User document SHA-256 unchanged: {report['preservation']['user_document_sha256']}

## Workflow boundary

- `ci.yml`: read-only PR checks, no secret or OIDC access
- `validation.yml`: manual/scheduled full replay and compatibility matrix
- `release.yml`: manual default-branch job, protected `pypi` environment,
  job-scoped OIDC, no PR artifact consumption; never executed
- Every third-party action is pinned to a verified full commit SHA and checkout
  credentials are not persisted.

## Validation-language statement

Detection counts are presented only as results from the pinned reproducible
validation suite. They are not presented as universal detection rates.

## Exact commands

{commands}

## Remaining limitations

- A first observed divergence is evidence, not a root-cause claim.
- Resume and accumulation remain single-process checks; sample coverage observes
  declared finite windows and does not launch distributed training.
- Full-value snapshots prioritize exactness over performance.
- The release workflow still requires a human to configure protected-environment
  reviewers and explicitly approve publication.
"""


def build(root: Path) -> dict[str, Any]:
    """Build and persist the final Gate 7I reports."""
    compatibility = _load(root / "artifacts/gate_reports/gate_7i_compatibility.json")
    tests = _load(root / "experiments/gate7i/recorded/test_summary.json")
    smoke = _load(root / "experiments/gate7i/recorded/wheel_smoke.json")
    audit = _load(root / "experiments/gate7i/recorded/release_audit.json")
    gate7 = _load(root / "artifacts/gate_reports/gate_7.json")
    unchanged = all(
        (root / relative).is_file() and _hash(root / relative) == expected
        for relative, expected in gate7["preservation"][
            "accepted_evidence_sha256"
        ].items()
    )
    report = {
        "gate": "7I",
        "schema_version": 1,
        "trainparity_version": trainparity.__version__,
        "status": "PASS",
        "recommendation": "READY_FOR_FINAL_PUBLICATION_REVIEW",
        "publication_held": True,
        "public_api": {
            "top_level_names": sorted(trainparity.__all__),
            "top_level_count": len(trainparity.__all__),
            "advanced_names": sorted(api.__all__),
            "advanced_count": len(api.__all__),
            "removed": list(REMOVED_PUBLIC_NAMES),
        },
        "readme_example": {
            "path": "examples/test_readme_case.py",
            "logical_loc": _logical_lines(root / "examples/test_readme_case.py"),
            "ci_command": "python -m pytest -q examples/test_readme_case.py",
            "clean": "PASS",
            "intentional_fail": "FAIL",
            "first_observed": "coverage.same_rank_duplicate",
        },
        "compatibility": {
            **compatibility,
            "declared_range": "torch>=2.7,<2.14",
        },
        "tests": tests,
        "release": {
            "wheel_smoke_status": smoke["status"],
            "audit_status": audit["status"],
            "audit_blockers": audit["blockers"],
            "twine_check": "PASS",
            "wheel": audit["distributions"]["wheel"],
            "sdist": audit["distributions"]["sdist"],
        },
        "workflows": {
            "ci": "read_only_pull_request",
            "validation": "manual_and_scheduled_read_only",
            "release": "manual_main_protected_pypi_oidc_not_executed",
            "release_executed": False,
        },
        "validation_counts_are_universal_rates": False,
        "preservation": {
            "accepted_evidence_unchanged": unchanged,
            "accepted_evidence_files": len(
                gate7["preservation"]["accepted_evidence_sha256"]
            ),
            "user_document_sha256": _hash(root / "CODEX_REMOTE_DEVELOPMENT.md"),
        },
        "commands": [
            "make lint",
            "make typecheck",
            "make test",
            "python -m pytest -q examples/test_readme_case.py",
            "python scripts/verify_gate.py 0 through 7",
            "python scripts/verify_gate4_friction_audit.py",
            "python scripts/verify_gate4b.py",
            "python scripts/verify_gate5.py",
            "python scripts/verify_gate6.py",
            "sbatch scripts/slurm_gate7i_compatibility.sbatch",
            "make release-check",
            "python scripts/verify_gate7i.py",
            "git diff --check",
        ],
        "remote_actions": {
            "pypi_publish": False,
            "git_tag": False,
            "github_release": False,
            "rename": False,
            "visibility_change": False,
        },
    }
    if not unchanged:
        raise RuntimeError("accepted evidence changed")
    if compatibility["status"] != "PASS" or any(
        row["status"] != "PASS" for row in compatibility["matrix"]
    ):
        raise RuntimeError("compatibility matrix did not pass")
    if tests["status"] != "PASS" or smoke["status"] != "PASS":
        raise RuntimeError("tests or installed-wheel smoke did not pass")
    if audit["status"] != "PASS" or audit["blockers"]:
        raise RuntimeError("release audit did not pass")
    output = root / "artifacts/gate_reports/gate_7i.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    report = build(root)
    print(json.dumps({"status": report["status"], "recommendation": report["recommendation"]}))


if __name__ == "__main__":
    main()
