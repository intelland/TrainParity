"""Verify the bounded Gate 7I release surface and review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import trainparity
from trainparity import api

TOP_LEVEL_API = {
    "check_resume",
    "check_accumulation",
    "audit_sample_coverage",
    "ExactlyOnce",
    "AtLeastOnce",
    "NoCrossRankOverlap",
    "ExpectedPadding",
    "ExactComparison",
    "ToleranceComparison",
    "Outcome",
    "__version__",
}
PUBLIC_FILES = (
    "README.md",
    "CHANGELOG.md",
    "MANIFEST.in",
    "pyproject.toml",
    "examples/test_readme_case.py",
    "src/trainparity/__init__.py",
    "src/trainparity/api.py",
    "docs/api.md",
    "docs/public-api-inventory.md",
    "docs/validation.md",
    "docs/design.md",
    "docs/comparison-with-traincheck.md",
    "docs/limitations.md",
    "docs/development-provenance.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "SECURITY.md",
    "docs/release-notes/0.1.0rc1.md",
    ".github/workflows/ci.yml",
    ".github/workflows/validation.yml",
    ".github/workflows/release.yml",
)
EXPECTED_BUNDLE = {
    *PUBLIC_FILES,
    "artifacts/gate_reports/gate_7i.json",
    "artifacts/gate_reports/gate_7i.md",
    "artifacts/gate_reports/gate_7i_compatibility.json",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Gate 7I verification failed: {message}")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_links(root: Path) -> int:
    checked = 0
    for relative in PUBLIC_FILES:
        path = root / relative
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            clean = target.split("#", 1)[0]
            if not clean:
                continue
            _require((path.parent / clean).resolve().exists(), f"broken link {relative}: {target}")
            checked += 1
    return checked


def _verify_workflows(root: Path) -> None:
    paths = sorted((root / ".github/workflows").glob("*.yml"))
    _require({path.name for path in paths} == {"ci.yml", "validation.yml", "release.yml"}, "workflow inventory")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        references = re.findall(r"uses:\s+[^@\s]+@([^\s]+)", text)
        _require(bool(references), f"workflow actions {path.name}")
        _require(all(re.fullmatch(r"[0-9a-f]{40}", value) for value in references), f"unpinned action {path.name}")
        _require(text.count("actions/checkout@") == text.count("persist-credentials: false"), f"checkout credentials {path.name}")
        _require("permissions:\n  contents: read" in text, f"default permissions {path.name}")
    ci = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    validation = (root / ".github/workflows/validation.yml").read_text(encoding="utf-8")
    release = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    _require("pull_request:" in ci and "id-token: write" not in ci and "secrets." not in ci, "PR isolation")
    _require("workflow_dispatch:" in validation and "schedule:" in validation, "validation triggers")
    _require("id-token: write" not in validation and "secrets." not in validation, "validation permissions")
    for phrase in (
        "github.ref == 'refs/heads/main'",
        "PYPI_ENVIRONMENT_PROTECTED",
        "name: pypi",
        "id-token: write",
        "gh-action-pypi-publish@",
        "group: trainparity-pypi-release",
        "cancel-in-progress: false",
    ):
        _require(phrase in release, f"release protection {phrase}")
    _require("download-artifact" not in release, "untrusted artifact consumption")
    _require(release.count("python -m build") == 1, "release builds exactly once")
    smoke = "- name: Smoke-test the exact wheel from outside the source tree"
    publish = "- name: Publish with PyPI Trusted Publishing"
    _require(smoke in release and release.index(smoke) < release.index(publish), "release smoke order")
    for phrase in (
        "wheels=(dist/*.whl)",
        'pip install --no-deps "${wheels[0]}"',
        'cd "${RUNNER_TEMP}"',
        "trainparity.__version__",
        "trainparity.quickstarts.resume",
        "trainparity.quickstarts.accumulation",
        "trainparity.quickstarts.sample_coverage",
        "packages-dir: dist/",
    ):
        _require(phrase in release, f"same-artifact release smoke {phrase}")


def verify(root: Path, *, fast: bool) -> dict[str, Any]:
    """Verify source contracts and, unless fast, recorded Gate 7I evidence."""
    _require(set(trainparity.__all__) == TOP_LEVEL_API, "top-level API")
    for name in ("PACKAGE_VERSION", "ExternalProcessEvidence", "SampleCoverageAuditor"):
        _require(name not in api.__all__, f"removed public name {name}")
    _require(all((root / relative).is_file() for relative in PUBLIC_FILES), "public file inventory")
    _require(not (root / "src/trainparity/cli.py").exists(), "CLI module removal")
    _require(not (root / "src/trainparity/__main__.py").exists(), "module CLI removal")
    readme = (root / "README.md").read_text(encoding="utf-8")
    first = readme.split("## Installed quickstarts", 1)[0]
    _require("img.shields.io/pypi/v/trainparity.svg" in readme, "published PyPI badge")
    _require(
        readme.index("## Install and run") < readme.index("## A complete integration"),
        "README install placement",
    )
    _require("Asking Codex" not in readme, "Codex paragraph")
    placeholder = re.search(r"^\s*(?:pass|\.\.\.)\s*$", first, re.MULTILINE)
    _require("class CoverageCase:" in first and placeholder is None, "complete first-screen case")
    _require(
        "python -m pytest -q --no-cov examples/test_readme_case.py" in first,
        "README pytest command",
    )
    _require("pip install trainparity==0.1.0rc1" in readme, "README PyPI install")
    relative_public_links = re.findall(
        r"\]\(((?:docs|examples)/[^)]+|[A-Z]+\.md|LICENSE)\)", readme
    )
    _require(not relative_public_links, "PyPI-safe README links")
    shipped_release_text = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in ("README.md", "CHANGELOG.md", "docs/release-notes/0.1.0rc1.md")
    )
    for phrase in (
        "Publication remains held",
        "not published to PyPI",
        "No publication workflow has been executed",
        "unpublished release candidate",
    ):
        _require(phrase not in shipped_release_text, f"publication-safe documentation: {phrase}")
    _verify_workflows(root)
    links = _verify_links(root)
    gate7 = _load(root / "artifacts/gate_reports/gate_7.json")
    for relative, expected in gate7["preservation"]["accepted_evidence_sha256"].items():
        _require(_hash(root / relative) == expected, f"accepted evidence {relative}")
    document_hash = _hash(root / "CODEX_REMOTE_DEVELOPMENT.md")
    allowed_document_hashes = {
        "6b532b94660949abec3c50bbe826d2154a797eeec744d3e5c91058dec9a96300",
        gate7["preservation"]["tracked_remote_development_sha256"],
    }
    _require(document_hash in allowed_document_hashes, "user document")
    if not fast:
        compatibility = _load(root / "artifacts/gate_reports/gate_7i_compatibility.json")
        _require(compatibility["status"] == "PASS", "compatibility status")
        _require(
            {row["torch"].split("+", 1)[0] for row in compatibility["matrix"]}
            == {"2.7.0", "2.10.0", "2.13.0"},
            "compatibility versions",
        )
        _require(all(row["status"] == "PASS" for row in compatibility["matrix"]), "compatibility rows")
        audit = _load(root / "experiments/gate7i/recorded/release_audit.json")
        _require(audit["status"] == "PASS" and not audit["blockers"], "release audit")
        report = _load(root / "artifacts/gate_reports/gate_7i.json")
        _require(report["status"] == "PASS" and report["publication_held"], "Gate 7I report")
        bundle = root / "artifacts/gate7i/trainparity-gate7i-human-review.zip"
        with zipfile.ZipFile(bundle) as archive:
            _require(set(archive.namelist()) == EXPECTED_BUNDLE, "real-path bundle inventory")
    return {
        "status": "PASS",
        "top_level_names": len(TOP_LEVEL_API),
        "linked_files_checked": links,
        "publication_held": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(root, fast=arguments.fast), sort_keys=True))


if __name__ == "__main__":
    main()
