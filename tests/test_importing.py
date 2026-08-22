from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from trainparity.importing import CaseImportError, load_accumulation_case, load_process_case
from trainparity.protocols import AccumulationCase, ProcessResumeCase

PROCESS_CASE_SPEC = "trainparity.quickstarts.resume:CleanCase"
ACCUMULATION_CASE_SPEC = "trainparity.quickstarts.accumulation:Case"


@pytest.mark.parametrize(
    ("loader", "spec", "protocol"),
    [
        (load_process_case, PROCESS_CASE_SPEC, ProcessResumeCase),
        (load_accumulation_case, ACCUMULATION_CASE_SPEC, AccumulationCase),
    ],
)
def test_current_loaders_match_their_protocols(
    loader: Callable[[str], object], spec: str, protocol: type[object]
) -> None:
    assert isinstance(loader(spec), protocol)


@pytest.mark.parametrize("loader", [load_process_case, load_accumulation_case])
@pytest.mark.parametrize(
    "spec",
    ["invalid", "trainparity:missing", "trainparity.version:add_report_metadata"],
)
def test_current_loaders_reject_invalid_targets(
    loader: Callable[[str], object], spec: str
) -> None:
    with pytest.raises(CaseImportError):
        loader(spec)


@pytest.mark.parametrize(
    ("loader_name", "spec", "expected_name"),
    [
        ("load_process_case", PROCESS_CASE_SPEC, "CleanCase"),
        ("load_accumulation_case", ACCUMULATION_CASE_SPEC, "Case"),
    ],
)
def test_current_cases_import_in_a_fresh_process(
    loader_name: str, spec: str, expected_name: str, tmp_path: Path
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"from trainparity.importing import {loader_name}; "
                f"case = {loader_name}({spec!r}); "
                "print(type(case).__name__)"
            ),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected_name
