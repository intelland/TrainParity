from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _documented_file(guide: str, marker: str) -> str:
    opening = f"```python\n# {marker}\n"
    start = guide.index(opening) + len("```python\n")
    end = guide.index("\n```", start)
    return guide[start:end] + "\n"


def test_external_resume_guide_documents_the_public_execution_contract() -> None:
    guide = (ROOT / "docs/external-resume-integration.md").read_text(encoding="utf-8")
    api = (ROOT / "docs/api.md").read_text(encoding="utf-8")

    for phase in (
        "baseline_a",
        "baseline_b",
        "candidate_split",
        "candidate_resume",
    ):
        assert f"`{phase}`" in guide
        assert f"`{phase}`" in api

    assert "before that child starts" in guide
    assert "before* launching the resumed child" in api
    assert 'return run_dir / "checkpoint.pt"' in guide
    assert "saved/models/<timestamp>/checkpoint.pth" in guide
    assert "candidate_resume/stderr.log under the preserved work_dir" in guide
    assert "model parameters and buffers" in guide
    assert "not an inferred root cause" in guide
    assert "roughly three normal-run" in guide


def test_external_resume_guide_contains_complete_user_files() -> None:
    guide = (ROOT / "docs/external-resume-integration.md").read_text(encoding="utf-8")

    assert "# project/train.py" in guide
    assert "# trainparity_case.py" in guide
    assert "# run_trainparity.py" in guide
    assert "class Case:" in guide
    assert "def command(" in guide
    assert "def checkpoint_path(" in guide
    assert "def observe_checkpoint(" in guide
    assert 'result = check_resume(' in guide
    assert "..." not in guide


def test_external_resume_guide_example_runs_as_copied(tmp_path: Path) -> None:
    guide = (ROOT / "docs/external-resume-integration.md").read_text(encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text(
        _documented_file(guide, "project/train.py"), encoding="utf-8"
    )
    (tmp_path / "trainparity_case.py").write_text(
        _documented_file(guide, "trainparity_case.py"), encoding="utf-8"
    )
    (tmp_path / "run_trainparity.py").write_text(
        _documented_file(guide, "run_trainparity.py"), encoding="utf-8"
    )

    completed = subprocess.run(
        [sys.executable, "run_trainparity.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "trainparity-report.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "PASS"
    assert (tmp_path / ".trainparity-runs/candidate_resume/stderr.log").is_file()
