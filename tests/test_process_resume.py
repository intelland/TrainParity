from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from trainparity.outcomes import Outcome
from trainparity.process_resume import ProcessResumeRunner

PREFIX = "trainparity.examples.process_cases:"


def test_process_runner_propagates_explicit_environment_without_reporting_values(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    report_path = tmp_path / "report.json"
    result = ProcessResumeRunner(temporary_root=scratch).run(
        PREFIX + "DeterministicProcessCase",
        report_path=report_path,
        environment={
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "TRAINPARITY_REQUIRE_TORCH_FORCE": "1",
            "TRAINPARITY_TEST_SECRET": "do-not-report-this-value",
        },
    )
    assert result.outcome is Outcome.PASS
    assert result.fresh_resume_processes_distinct
    assert len(result.processes) == 4
    assert result.processes[2].pid != result.processes[3].pid
    assert result.timing_seconds is not None
    for phase in (
        "single_normal_run",
        "baseline_self_consistency",
        "candidate_save_exit_new_process_load_resume",
        "snapshot_capture",
        "serialization",
        "comparison",
        "total_wall",
        "end_to_end_multiplier",
    ):
        assert result.timing_seconds[phase] >= 0
    assert result.propagated_environment_keys == (
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        "TRAINPARITY_REQUIRE_TORCH_FORCE",
        "TRAINPARITY_TEST_SECRET",
    )
    report = report_path.read_text(encoding="utf-8")
    assert "do-not-report-this-value" not in report
    assert json.loads(report)["outcome"] == "PASS"
    assert list(scratch.iterdir()) == []


def test_staged_checkpoint_fault_reports_first_observed_path(tmp_path: Path) -> None:
    def fault(path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint["scheduler"]["last_epoch"] -= 1
        torch.save(checkpoint, path)

    result = ProcessResumeRunner(temporary_root=tmp_path / "scratch").run(
        PREFIX + "DeterministicProcessCase",
        staged_checkpoint_hook=fault,
    )
    assert result.outcome is Outcome.FAIL
    assert result.first_divergent_step == 4
    assert result.primary_difference is not None
    assert result.primary_difference.path == "scheduler.last_epoch"
    assert result.fresh_resume_processes_distinct


def test_baseline_nondeterminism_abstains(tmp_path: Path) -> None:
    result = ProcessResumeRunner(temporary_root=tmp_path).run(
        PREFIX + "NondeterministicProcessCase"
    )
    assert result.outcome is Outcome.ABSTAIN
    assert "self-consistency" in result.message


@pytest.mark.parametrize(
    ("case", "timeout", "outcome"),
    [
        ("ErrorProcessCase", 10.0, Outcome.ERROR),
        ("SlowProcessCase", 0.1, Outcome.ERROR),
        ("UnsupportedProcessCase", 10.0, Outcome.ABSTAIN),
        ("MissingCase", 10.0, Outcome.ERROR),
    ],
)
def test_process_runner_preserves_non_pass_outcomes(
    case: str, timeout: float, outcome: Outcome, tmp_path: Path
) -> None:
    result = ProcessResumeRunner(timeout=timeout, temporary_root=tmp_path).run(PREFIX + case)
    assert result.outcome is outcome


def test_process_runner_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ProcessResumeRunner(timeout=0)
    result = ProcessResumeRunner(temporary_root=tmp_path).run(
        PREFIX + "DeterministicProcessCase",
        environment={"BAD=KEY": "value"},
    )
    assert result.outcome is Outcome.ERROR
