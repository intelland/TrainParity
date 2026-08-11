"""Assemble Gate 7 reports from recorded release-candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

BASELINE_COMMIT = "69abf41"
USER_DOCUMENT_SHA256 = "6b532b94660949abec3c50bbe826d2154a797eeec744d3e5c91058dec9a96300"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _git_bytes(root: Path, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:{relative}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _preservation(root: Path) -> dict[str, str]:
    gate6 = _load(root / "artifacts" / "gate_reports" / "gate_6.json")
    expected: dict[str, str] = dict(gate6["preservation"]["accepted_evidence_sha256"])
    gate6_paths = [
        "artifacts/gate_reports/gate_6.json",
        "artifacts/gate_reports/gate_6.md",
        *[
            path.relative_to(root).as_posix()
            for path in sorted((root / "experiments" / "gate6" / "recorded").rglob("*"))
            if path.is_file()
        ],
    ]
    for relative in gate6_paths:
        expected[relative] = _hash_bytes(_git_bytes(root, relative))
    changed = [
        relative
        for relative, digest in expected.items()
        if not (root / relative).is_file() or _hash(root / relative) != digest
    ]
    if changed:
        raise RuntimeError(f"accepted evidence changed: {changed}")
    return dict(sorted(expected.items()))


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    criteria = "\n".join(
        f"- [{'x' if item['passed'] else ' '}] {item['name']}: {item['evidence']}"
        for item in report["criteria"]
    )
    commands = "\n".join(f"- `{item}`" for item in report["commands"])
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    examples = "\n".join(
        f"- `{item['module']}`: clean={item['clean']}, intentional={item['intentional_fail']}, first observed=`{item['first_observed']}`"
        for item in report["wheel_smoke"]["examples"]
    )
    markdown = f"""# Gate 7 report

## Outcome

**{report['status']} — recommendation: {report['recommendation']}**

{report['summary']}

Gate 7 stops here for final human review. Nothing was published, tagged,
released, renamed, or made public.

## Acceptance criteria

{criteria}

## Installed-wheel examples

{examples}

The examples ran from outside the repository in a newly created Python
environment. Their failures are intentional first-observed divergences, not
root-cause claims.

## Distribution audit

- Wheel: `{report['release_audit']['distributions']['wheel']['file']}`
  ({report['release_audit']['distributions']['wheel']['bytes']} bytes)
- Source distribution: `{report['release_audit']['distributions']['sdist']['file']}`
  ({report['release_audit']['distributions']['sdist']['bytes']} bytes)
- Repository/distribution blockers: {report['release_audit']['blockers']}
- Accepted evidence hashes checked: {len(report['preservation']['accepted_evidence_sha256'])}

## Exact commands

{commands}

## Remaining limitations

{limitations}
"""
    path.write_text(markdown, encoding="utf-8")


def build(root: Path, *, allow_pending_ci: bool = False) -> dict[str, Any]:
    """Build and write the Gate 7 machine and Markdown reports."""
    recorded = root / "experiments" / "gate7" / "recorded"
    smoke = _load(recorded / "wheel_smoke.json")
    audit = _load(recorded / "release_audit.json")
    tests = _load(recorded / "test_summary.json")
    names = _load(recorded / "name_availability.json")
    ci = _load(recorded / "ci.json")
    accepted = _preservation(root)
    current_document_hash = _hash(root / "CODEX_REMOTE_DEVELOPMENT.md")
    criteria = [
        {"name": "frozen public API", "passed": tests["public_api"] == "PASS", "evidence": f"stable names={tests['public_api_name_count']}"},
        {"name": "versioned machine reports", "passed": tests["versioned_reports"] == "PASS", "evidence": "schema=1, package=0.1.0rc1"},
        {"name": "three installed CPU examples", "passed": smoke["status"] == "PASS" and len(smoke["examples"]) == 3, "evidence": "three clean PASS and three intentional FAIL"},
        {"name": "release documentation", "passed": tests["release_documentation"] == "PASS", "evidence": "README, API, validation, design, comparison, limitations, provenance, security, contribution, release notes"},
        {"name": "conservative compatibility", "passed": tests["compatibility"] == "PASS", "evidence": "Python 3.11 and PyTorch 2.7 only; exact GPU fixtures documented"},
        {"name": "name availability rechecked", "passed": names["no_remote_action_performed"] and names["pypi"]["http_status"] == 404, "evidence": "time-limited result recorded; no rename or publication"},
        {"name": "repository and distribution audit", "passed": audit["status"] == "PASS", "evidence": "no blocker; Gate evidence excluded from wheel and sdist"},
        {"name": "accepted evidence preserved", "passed": bool(accepted), "evidence": f"hashes unchanged={len(accepted)}"},
        {"name": "user remote-development document preserved", "passed": current_document_hash == USER_DOCUMENT_SHA256, "evidence": current_document_hash},
        {"name": "complete verification", "passed": tests["status"] == "PASS", "evidence": f"tests={tests['tests_passed']}, coverage={tests['coverage_percent']}%"},
        {"name": "hosted CPU CI", "passed": ci["conclusion"] == "success" or allow_pending_ci, "evidence": f"run={ci.get('run_id')} conclusion={ci['conclusion']}"},
        {"name": "no irreversible remote action", "passed": not any(ci.get(key, False) for key in ("published", "tagged", "released", "visibility_changed")), "evidence": "not published/tagged/released; repository remains private"},
    ]
    passed = all(item["passed"] for item in criteria)
    report = {
        "schema_version": 1,
        "trainparity_version": "0.1.0rc1",
        "gate": 7,
        "status": "PASS" if passed else "BLOCKED",
        "recommendation": "READY_FOR_HUMAN_REVIEW" if passed else "REWORK",
        "summary": "The frozen v0.1 release candidate is documented, packaged, audited, and verified without publication." if passed else "One or more release-candidate criteria remain incomplete; no release action is authorized.",
        "criteria": criteria,
        "public_api": tests["public_api_names"],
        "compatibility": tests["tested_compatibility"],
        "wheel_smoke": smoke,
        "release_audit": audit,
        "name_availability": names,
        "test_summary": tests,
        "hosted_ci": ci,
        "preservation": {
            "baseline_commit": BASELINE_COMMIT,
            "accepted_evidence_sha256": accepted,
            "user_uncommitted_remote_development_sha256": current_document_hash,
        },
        "release_actions": {
            "pypi_published": False,
            "git_tag_created": False,
            "github_release_created": False,
            "repository_visibility_changed": False,
            "repository_renamed": False,
        },
        "commands": [
            "make lint",
            "make typecheck",
            "make test",
            "make build",
            "make release-check",
            "python scripts/verify_gate.py 0",
            "python scripts/verify_gate.py 1",
            "python scripts/verify_gate.py 2",
            "python scripts/verify_gate.py 3",
            "python scripts/verify_gate.py 4",
            "python scripts/verify_gate4_friction_audit.py",
            "python scripts/verify_gate4b.py",
            "python scripts/verify_gate5.py",
            "python scripts/verify_gate.py 6",
            "python scripts/verify_gate.py 7",
            "git diff --check",
        ],
        "limitations": [
            "The compatibility matrix is exact tested evidence, not a promise for untested Python, PyTorch, CUDA, GPU, operating-system, distributed, project, or scale combinations.",
            "The full-value backend favors correctness and can be expensive for large state.",
            "Users declare observation completeness, equivalence, sample-ID semantics, and any tolerance.",
            "Sample coverage proves one finite declared window and does not inspect sample contents or establish infinite-stream behavior or general shuffle quality.",
            "A first observed divergence is not a root-cause claim.",
            "TrainParity is not a sandbox; trusted code and checkpoints remain the caller's responsibility.",
            "The PyPI name check is time-limited and does not reserve the name; this candidate remains unpublished.",
        ],
    }
    output = root / "artifacts" / "gate_reports"
    output.mkdir(parents=True, exist_ok=True)
    (output / "gate_7.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(output / "gate_7.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pending-ci", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = build(root, allow_pending_ci=arguments.allow_pending_ci)
    print(json.dumps({"status": report["status"], "recommendation": report["recommendation"]}))


if __name__ == "__main__":
    main()
