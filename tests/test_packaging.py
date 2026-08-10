from __future__ import annotations

import tomllib
from pathlib import Path


def test_production_dependencies_are_minimal_and_documented() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["dependencies"] == ["torch>=2.5"]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "sole production dependency is `torch>=2.5`" in readme


def test_forbidden_runtime_dependencies_are_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    forbidden = {"cloudpickle", "openai", "langchain", "ray", "fastapi", "flask"}
    names = {dependency.split("[", 1)[0].split(">", 1)[0] for dependency in project["dependencies"]}
    assert names.isdisjoint(forbidden)

