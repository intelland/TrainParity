"""Verify the current supported repository and publication surface."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import trainparity
from trainparity import api
from trainparity.version import MACHINE_REPORT_SCHEMA_VERSION, PACKAGE_VERSION

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
ADVANCED_API = {
    "MACHINE_REPORT_SCHEMA_VERSION",
    "AccumulationCase",
    "AccumulationExecutionPlan",
    "AccumulationResult",
    "AtLeastOnce",
    "ComparisonPolicy",
    "Difference",
    "ExactComparison",
    "ExactlyOnce",
    "ExpectedPadding",
    "LossAccounting",
    "NoCrossRankOverlap",
    "Outcome",
    "ProcessExecutionPlan",
    "ProcessResumeCase",
    "ProcessResumeResult",
    "SampleAnomaly",
    "SampleCoverageResult",
    "SampleObservation",
    "SampleViolation",
    "ToleranceComparison",
    "TrainingState",
    "audit_rank_iterables",
    "audit_sample_coverage",
    "check_accumulation",
    "check_resume",
}
PUBLIC_FILES = (
    "README.md",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "SECURITY.md",
    "pyproject.toml",
    "examples/test_readme_case.py",
    "src/trainparity/__init__.py",
    "src/trainparity/api.py",
    "src/trainparity/version.py",
    "docs/api.md",
    "docs/comparison-with-traincheck.md",
    "docs/design.md",
    "docs/external-resume-integration.md",
    "docs/limitations.md",
    "docs/public-api-inventory.md",
    "docs/validation.md",
    "docs/release-notes/0.1.0.md",
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    ".github/workflows/validation.yml",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Release-surface verification failed: {message}")


def _verify_links(root: Path) -> int:
    checked = 0
    link_pattern = re.compile(
        r"!?"
        + re.escape("[")
        + r"[^]]*"
        + re.escape("](")
        + r"([^)]+)"
        + re.escape(")")
    )
    for relative in PUBLIC_FILES:
        path = root / relative
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            clean = target.split("#", 1)[0]
            if not clean:
                continue
            _require(
                (path.parent / clean).resolve().exists(),
                f"broken link {relative}: {target}",
            )
            checked += 1
    return checked


def _verify_workflows(root: Path) -> None:
    workflow_root = root / ".github/workflows"
    paths = sorted(workflow_root.glob("*.yml"))
    _require(
        {path.name for path in paths}
        == {"ci.yml", "validation.yml", "release.yml"},
        "workflow inventory",
    )
    contents = {
        path.name: path.read_text(encoding="utf-8") for path in paths
    }
    for name, text in contents.items():
        references = re.findall(r"uses:\s+[^@\s]+@([^\s]+)", text)
        _require(bool(references), f"workflow actions {name}")
        _require(
            all(re.fullmatch(r"[0-9a-f]{40}", value) for value in references),
            f"unpinned action {name}",
        )
        _require(
            text.count("actions/checkout@")
            == text.count("persist-credentials: false"),
            f"checkout credentials {name}",
        )
        _require("permissions:\n  contents: read" in text, f"permissions {name}")

    ci = contents["ci.yml"]
    validation = contents["validation.yml"]
    release = contents["release.yml"]
    _require(
        "pull_request:" in ci
        and "id-token: write" not in ci
        and "secrets." not in ci,
        "PR isolation",
    )
    _require(
        "python scripts/verify_release_surface.py" in ci,
        "current release-surface verifier",
    )
    _require("verify_gate" not in ci, "legacy verifier in CI")

    _require(
        "workflow_dispatch:" in validation and 'cron: "17 3 * * 1"' in validation,
        "validation triggers",
    )
    _require(
        "id-token: write" not in validation and "secrets." not in validation,
        "validation permissions",
    )
    validation_jobs = validation.split("\njobs:\n", 1)
    _require(len(validation_jobs) == 2, "validation jobs section")
    _require(
        set(re.findall(r"(?m)^  ([a-z][a-z0-9-]*):$", validation_jobs[1]))
        == {"compatibility"},
        "validation job inventory",
    )
    for phrase in (
        'torch: ["2.7.0", "2.10.0", "2.13.0"]',
        "https://download.pytorch.org/whl/cpu",
        "python -m build --wheel",
        'cd "$RUNNER_TEMP"',
        "scripts/compatibility_check.py",
    ):
        _require(phrase in validation, f"compatibility workflow {phrase}")
    _require("verify_gate" not in validation, "legacy verifier in validation")

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
    _require("download-artifact" not in release, "PR artifact consumption")
    _require(release.count("python -m build") == 1, "release builds once")
    smoke = "- name: Smoke-test the exact wheel from outside the source tree"
    publish = "- name: Publish with PyPI Trusted Publishing"
    _require(
        smoke in release and release.index(smoke) < release.index(publish),
        "release smoke order",
    )
    for phrase in (
        "wheels=(dist/*.whl)",
        'pip install --no-deps "${wheels[0]}"',
        'cd "${RUNNER_TEMP}"',
        "trainparity.__version__",
        "MACHINE_REPORT_SCHEMA_VERSION == 2",
        "trainparity.quickstarts.resume",
        "trainparity.quickstarts.accumulation",
        "trainparity.quickstarts.sample_coverage",
        "packages-dir: dist/",
    ):
        _require(phrase in release, f"same-artifact release smoke {phrase}")


def verify(root: Path) -> dict[str, object]:
    """Verify current public APIs, docs, workflows, and release boundaries."""
    _require(set(trainparity.__all__) == TOP_LEVEL_API, "top-level API")
    _require(set(api.__all__) == ADVANCED_API, "advanced API")
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    project_version = project["version"]
    _require(isinstance(project_version, str), "project version type")
    _require(
        trainparity.__version__ == PACKAGE_VERSION == project_version,
        "package version",
    )
    _require(MACHINE_REPORT_SCHEMA_VERSION == 2, "machine-report schema")
    _require(
        all((root / relative).is_file() for relative in PUBLIC_FILES),
        "public file inventory",
    )
    _require(not (root / "src/trainparity/cli.py").exists(), "CLI module")
    _require(not (root / "src/trainparity/__main__.py").exists(), "module CLI")

    readme = (root / "README.md").read_text(encoding="utf-8")
    _require("img.shields.io/pypi/v/trainparity.svg" in readme, "PyPI badge")
    _require("pip install trainparity" in readme, "generic install")
    _require("pip install trainparity==0.1.0" in readme, "versioned install")
    for module in ("resume", "accumulation", "sample_coverage"):
        _require(
            f"python -m trainparity.quickstarts.{module}" in readme,
            f"README {module} quickstart",
        )
    for target in (
        "docs/api.md",
        "docs/validation.md",
        "docs/design.md",
        "docs/limitations.md",
        "SECURITY.md",
        "CONTRIBUTING.md",
    ):
        _require(
            f"https://github.com/intelland/TrainParity/blob/main/{target}"
            in readme,
            f"README public navigation {target}",
        )
    relative_public_links = re.findall(
        r"\]\(((?:docs|examples)/[^)]+|[A-Z]+\.md|LICENSE)\)",
        readme,
    )
    _require(not relative_public_links, "PyPI-safe README links")
    shipped_release_text = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "README.md",
            "CHANGELOG.md",
            "docs/release-notes/0.1.0.md",
        )
    )
    for phrase in (
        "Publication remains held",
        "not published to PyPI",
        "No publication workflow has been executed",
        "unpublished release candidate",
    ):
        _require(
            phrase not in shipped_release_text,
            f"publication-safe documentation: {phrase}",
        )

    _verify_workflows(root)
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    _require("verify-gate-" not in makefile, "legacy Make targets")
    links = _verify_links(root)
    return {
        "status": "PASS",
        "package_version": PACKAGE_VERSION,
        "schema_version": MACHINE_REPORT_SCHEMA_VERSION,
        "top_level_names": len(TOP_LEVEL_API),
        "advanced_names": len(ADVANCED_API),
        "linked_files_checked": links,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(root), sort_keys=True))


if __name__ == "__main__":
    main()
