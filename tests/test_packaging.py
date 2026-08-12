from __future__ import annotations

import tomllib
from pathlib import Path


def test_production_dependencies_are_minimal_and_documented() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["dependencies"] == ["torch>=2.7,<2.14"]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "PyTorch 2.7.0, 2.10.0, and 2.13.0" in readme


def test_forbidden_runtime_dependencies_are_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    forbidden = {"cloudpickle", "openai", "langchain", "ray", "fastapi", "flask"}
    names = {dependency.split("[", 1)[0].split(">", 1)[0] for dependency in project["dependencies"]}
    assert names.isdisjoint(forbidden)


def test_release_candidate_metadata_is_conservative() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["version"] == "0.1.0rc2"
    assert project["requires-python"] == ">=3.11,<3.12"
    assert "Programming Language :: Python :: 3.12" not in project["classifiers"]


def test_external_resume_guide_is_included_in_source_distributions() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include docs/external-resume-integration.md" in manifest.splitlines()
