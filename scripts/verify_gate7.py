"""Verify the frozen Gate 7 release-candidate report and boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any

BANNED_README_WORDS = (
    "robust",
    "seamless",
    "intelligent",
    "enterprise-ready",
    "battle-tested",
    "revolutionary",
    "blazing-fast",
    "production-grade",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Gate 7 verification failed: {message}")


def verify(root: Path, allow_pending_ci: bool = False) -> dict[str, Any]:
    """Verify release reports, documentation, archives, and preservation."""
    report = _load(root / "artifacts" / "gate_reports" / "gate_7.json")
    markdown = (root / "artifacts" / "gate_reports" / "gate_7.md").read_text(encoding="utf-8")
    _require(report["gate"] == 7, "gate number")
    _require(report["status"] == "PASS", "report status")
    if not allow_pending_ci:
        _require(report["hosted_ci"]["conclusion"] == "success", "hosted CI")
    _require(report["schema_version"] == 1 and report["trainparity_version"] == "0.1.0rc1", "report metadata")
    _require(all(report["release_actions"][key] is False for key in report["release_actions"]), "remote release action")
    readme = (root / "README.md").read_text(encoding="utf-8")
    readme_lower = readme.lower()
    _require(not any(word in readme_lower for word in BANNED_README_WORDS), "README promotional language")
    _require(readme.count("![") == 4, "top badge count")
    _require(readme.index("## Reproducible validation suite") < len(readme) // 2, "validation matrix placement")
    for phrase in (
        "not a universal bug detector",
        "does not invoke an LLM at runtime",
        "first observed divergence",
        "semantic samples must not share",
        "JSON `null`, never worker 0",
        "one finite observation window",
    ):
        _require(phrase in readme, f"README phrase: {phrase}")
    for module in ("resume", "accumulation", "sample_coverage"):
        command = f"python -m trainparity.quickstarts.{module}"
        _require(command in readme, f"README command {command}")
    required_files = (
        "CHANGELOG.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/api.md",
        "docs/validation.md",
        "docs/design.md",
        "docs/comparison-with-traincheck.md",
        "docs/limitations.md",
        "docs/development-provenance.md",
        "docs/release-notes/0.1.0rc1.md",
        "docs/assets/social-preview.svg",
        "docs/assets/social-preview.png",
        "docs/launch/pytorch-forums.md",
        "docs/launch/show-hn.md",
        "docs/launch/reddit.md",
    )
    _require(all((root / relative).is_file() for relative in required_files), "release document inventory")
    for relative in required_files[-3:]:
        launch = (root / relative).read_text(encoding="utf-8").lower()
        _require("feedback" in launch and "stars" not in launch, f"launch feedback {relative}")
    smoke = report["wheel_smoke"]
    _require(smoke["status"] == "PASS" and not smoke["source_tree_imported"], "wheel smoke")
    _require(len(smoke["examples"]) == 3, "wheel example count")
    _require(all(item["clean"] == "PASS" and item["intentional_fail"] == "FAIL" for item in smoke["examples"]), "wheel example outcomes")
    historical_audit = report["release_audit"]
    _require(
        historical_audit["status"] == "PASS" and not historical_audit["blockers"],
        "historical release audit",
    )
    wheel = root / "dist" / historical_audit["distributions"]["wheel"]["file"]
    sdist = root / "dist" / historical_audit["distributions"]["sdist"]["file"]
    audit = historical_audit
    historical_artifacts_present = wheel.is_file() and sdist.is_file()
    historical_artifacts_match = historical_artifacts_present and (
        _hash(wheel) == historical_audit["distributions"]["wheel"]["sha256"]
        and _hash(sdist) == historical_audit["distributions"]["sdist"]["sha256"]
    )
    if not historical_artifacts_match:
        # Gate 7I deliberately rebuilds distributions after hardening release-facing
        # metadata. Keep the accepted Gate 7 audit immutable, but require the current
        # artifacts to pass the same archive audit before replaying this verifier.
        current_audit_path = (
            root / "experiments" / "gate7i" / "recorded" / "release_audit.json"
        )
        _require(current_audit_path.is_file(), "current release audit after Gate 7I rebuild")
        audit = _load(current_audit_path)
        _require(
            audit["status"] == "PASS"
            and not audit["blockers"]
            and audit["trainparity_version"] == report["trainparity_version"],
            "current release audit after Gate 7I rebuild",
        )
        wheel = root / "dist" / audit["distributions"]["wheel"]["file"]
        sdist = root / "dist" / audit["distributions"]["sdist"]["file"]
    _require(_hash(wheel) == audit["distributions"]["wheel"]["sha256"], "wheel hash")
    _require(_hash(sdist) == audit["distributions"]["sdist"]["sha256"], "sdist hash")
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
    _require(not any("trainparity/examples/" in name for name in wheel_names), "experiment examples in wheel")
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = [member.name for member in archive.getmembers()]
    _require(not any("/experiments/" in name or "/artifacts/" in name for name in sdist_names), "Gate evidence in sdist")
    for relative, expected in report["preservation"]["accepted_evidence_sha256"].items():
        path = root / relative
        _require(path.is_file() and _hash(path) == expected, f"preservation {relative}")
    document_hash = _hash(root / "CODEX_REMOTE_DEVELOPMENT.md")
    allowed_document_hashes = {
        report["preservation"]["user_uncommitted_remote_development_sha256"]
    }
    if allow_pending_ci:
        allowed_document_hashes.add(
            report["preservation"]["tracked_remote_development_sha256"]
        )
    _require(document_hash in allowed_document_hashes, "user document")
    _require("Nothing was published, tagged" in markdown, "Markdown remote boundary")
    return {
        "status": "PASS",
        "recommendation": "READY_FOR_HUMAN_REVIEW",
        "examples": 3,
        "preserved_files": len(report["preservation"]["accepted_evidence_sha256"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pending-ci", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(root, arguments.allow_pending_ci), sort_keys=True))


if __name__ == "__main__":
    main()
