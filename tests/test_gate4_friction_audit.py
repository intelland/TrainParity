from __future__ import annotations

import ast
from pathlib import Path

from experiments.gate4.friction.run_audit import PROJECTS, _logical_lines

ROOT = Path(__file__).resolve().parents[1]


def test_user_files_are_self_contained_and_separated() -> None:
    common = ROOT / "experiments/gate4/friction/user_files/trainparity_clean_resume.py"
    assert _logical_lines(common) > 0
    for project in PROJECTS:
        directory = ROOT / "experiments/gate4/friction/user_files" / project["name"]
        files = {path.name for path in directory.glob("*.py")}
        assert files == {"trainparity_adapter.py", "trainparity_project_glue.py"}
        adapter = (directory / "trainparity_adapter.py").read_text(encoding="utf-8")
        glue = (directory / "trainparity_project_glue.py").read_text(encoding="utf-8")
        ast.parse(adapter)
        ast.parse(glue)
        assert "experiments.gate4" not in adapter
        assert "experiments.gate4" not in glue


def test_closer_handwritten_test_covers_required_state() -> None:
    path = ROOT / "experiments/gate4/friction/handwritten_fresh_resume.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    for component in ('"model"', '"optimizer"', '"scheduler"', '"rng"', '"torch_cpu"'):
        assert component in source
    assert "subprocess.run" in source
    assert "first observed divergence" in source


def test_existing_weak_comparator_remains_twelve_lines() -> None:
    assert _logical_lines(ROOT / "experiments/gate4/handwritten.py") == 12
