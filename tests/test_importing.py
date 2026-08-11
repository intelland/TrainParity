from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from trainparity.importing import CaseImportError, load_case
from trainparity.protocols import ResumeCase

CASE_SPEC = "trainparity.examples.resume_cases:CorrectResumeCase"


def test_load_case_matches_protocol() -> None:
    case = load_case(CASE_SPEC)
    assert isinstance(case, ResumeCase)


@pytest.mark.parametrize(
    "spec", ["invalid", "trainparity:missing", "trainparity.version:add_report_metadata"]
)
def test_load_case_rejects_invalid_targets(spec: str) -> None:
    with pytest.raises(CaseImportError):
        load_case(spec)


def test_case_imports_in_fresh_process(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from trainparity.importing import load_case; "
                f"case = load_case({CASE_SPEC!r}); "
                "print(type(case).__name__)"
            ),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "CorrectResumeCase"
