from __future__ import annotations

import re
import tomllib
from pathlib import Path

import trainparity
from trainparity import api

ROOT = Path(__file__).resolve().parents[1]
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


def test_top_level_and_advanced_api_boundaries() -> None:
    assert set(trainparity.__all__) == TOP_LEVEL_API
    assert len(api.__all__) == 26
    for removed in ("PACKAGE_VERSION", "ExternalProcessEvidence", "SampleCoverageAuditor"):
        assert removed not in trainparity.__all__
        assert removed not in api.__all__
    assert {
        "ProcessExecutionPlan",
        "ProcessResumeCase",
        "AccumulationCase",
        "TrainingState",
        "LossAccounting",
        "SampleObservation",
    } <= set(api.__all__)


def test_release_metadata_has_real_maintainer_and_no_console_script() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["authors"] == [
        {"name": "Zhou Xianhao", "email": "593403766@qq.com"}
    ]
    assert project["maintainers"] == project["authors"]
    assert "Development Status :: 3 - Alpha" in project["classifiers"]
    assert "scripts" not in project
    assert not (ROOT / "src/trainparity/cli.py").exists()
    assert not (ROOT / "src/trainparity/__main__.py").exists()


def test_readme_uses_the_complete_ci_case_before_quickstarts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    example = (ROOT / "examples/test_readme_case.py").read_text(encoding="utf-8")
    logical_lines = sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in example.splitlines()
    )
    first = readme.split("## Installed quickstarts", 1)[0]
    assert 20 <= logical_lines <= 30
    assert f"{logical_lines}-logical-line pytest case" in first
    assert "img.shields.io/pypi" not in readme
    assert "Asking Codex" not in readme
    assert "does not invoke an LLM at runtime" in first
    assert "class CoverageCase:" in first
    assert "coverage.same_rank_duplicate" in first
    assert not re.search(r"^\s*(?:pass|\.\.\.)\s*$", first, re.MULTILINE)
    assert "pytest -q examples/test_readme_case.py" in first
    for module in ("resume", "accumulation", "sample_coverage"):
        assert f"python -m trainparity.quickstarts.{module}" in readme


def test_workflows_are_split_pinned_and_least_privilege() -> None:
    workflows = ROOT / ".github/workflows"
    assert {path.name for path in workflows.glob("*.yml")} == {
        "ci.yml",
        "validation.yml",
        "release.yml",
    }
    contents = {
        path.name: path.read_text(encoding="utf-8") for path in workflows.glob("*.yml")
    }
    for text in contents.values():
        uses = re.findall(r"uses:\s+[^@\s]+@([^\s]+)", text)
        assert uses and all(re.fullmatch(r"[0-9a-f]{40}", value) for value in uses)
        assert text.count("actions/checkout@") == text.count("persist-credentials: false")
        assert "permissions:\n  contents: read" in text
    assert "pull_request:" in contents["ci.yml"]
    assert "id-token: write" not in contents["ci.yml"]
    assert "secrets." not in contents["ci.yml"]
    assert "workflow_dispatch:" in contents["validation.yml"]
    assert "schedule:" in contents["validation.yml"]
    assert "id-token: write" not in contents["validation.yml"]
    release = contents["release.yml"]
    assert "workflow_dispatch:" in release
    assert "github.ref == 'refs/heads/main'" in release
    assert "PYPI_ENVIRONMENT_PROTECTED" in release
    assert "name: pypi" in release
    assert "id-token: write" in release
    assert "gh-action-pypi-publish@" in release
    assert "download-artifact" not in release


def test_validation_and_coverage_boundary_are_documented() -> None:
    validation = (ROOT / "docs/validation.md").read_text(encoding="utf-8")
    assert "e08ff9257ed18d8d805304e32ba85a44553195fc" in validation
    assert "subprocess coverage" in validation.lower()
    assert "not a universal detection rate" in " ".join(validation.split())
