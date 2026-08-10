from __future__ import annotations

from pathlib import Path

import pytest

from trainparity.assertions import assert_resume_equivalent
from trainparity.outcomes import Outcome
from trainparity.results import ResumeResult
from trainparity.runner import ResumeRunner

PREFIX = "trainparity.examples.gate3_cases:"


def run_case(name: str, tmp_path: Path, *, timeout: float = 60.0) -> ResumeResult:
    return ResumeRunner(timeout=timeout).run(PREFIX + name, work_dir=tmp_path)


def test_clean_case_crosses_real_process_and_checkpoint(tmp_path: Path) -> None:
    result = run_case("DeterministicCase", tmp_path)
    assert result.outcome is Outcome.PASS
    assert result.last_matching_step == 4
    assert result.pre_save is not None and result.post_load is not None
    assert result.pre_save.pid != result.post_load.pid
    assert result.checkpoint_path is not None and Path(result.checkpoint_path).is_file()
    assert_resume_equivalent(result)


@pytest.mark.parametrize(
    ("case", "step", "path_prefix"),
    [
        ("MissingModelCase", 2, "model"),
        ("MissingOptimizerCase", 2, "optimizer"),
        ("MissingSchedulerCase", 2, "optimizer"),
        ("MissingPythonRngCase", 2, "rng.python"),
        ("CursorOffsetCase", 3, "batch.sample_ids"),
        ("StepOffByOneCase", 2, "step"),
        ("MissingHiddenGlobalCase", 2, "extra.hidden_module_counter"),
    ],
)
def test_faults_report_first_observed_divergence(
    case: str, step: int, path_prefix: str, tmp_path: Path
) -> None:
    result = run_case(case, tmp_path)
    assert result.outcome is Outcome.FAIL
    assert result.first_divergent_step == step
    assert result.phase == "completed_training_step"
    assert result.primary_difference is not None
    assert result.primary_difference.path.startswith(path_prefix)
    assert result.all_differences
    with pytest.raises(AssertionError):
        assert_resume_equivalent(result)


def test_baseline_nondeterminism_abstains(tmp_path: Path) -> None:
    result = run_case("NondeterministicCase", tmp_path)
    assert result.outcome is Outcome.ABSTAIN
    assert "self-consistency" in result.message


def test_missing_data_identity_abstains(tmp_path: Path) -> None:
    result = run_case("MissingBatchIdentityCase", tmp_path)
    assert result.outcome is Outcome.ABSTAIN


@pytest.mark.parametrize("case", ["ChildExceptionCase", "CorruptCheckpointCase", "MissingCheckpointCase"])
def test_runner_and_checkpoint_failures_are_errors(case: str, tmp_path: Path) -> None:
    result = run_case(case, tmp_path)
    assert result.outcome is Outcome.ERROR


def test_import_failure_is_error(tmp_path: Path) -> None:
    result = ResumeRunner().run("not_a_real_module:Case", work_dir=tmp_path)
    assert result.outcome is Outcome.ERROR


def test_timeout_is_error(tmp_path: Path) -> None:
    result = run_case("SlowCase", tmp_path, timeout=0.2)
    assert result.outcome is Outcome.ERROR
    assert "timeout" in result.message


def test_invalid_runner_arguments() -> None:
    with pytest.raises(ValueError):
        ResumeRunner(timeout=0)
    with pytest.raises(ValueError):
        ResumeRunner().run(PREFIX + "DeterministicCase", total_steps=2, split_step=2)
