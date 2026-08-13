from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from trainparity import ExactComparison, ToleranceComparison, check_resume
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


def _small_staged_model_nudge(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["model"]["weight"] += 1e-5
    torch.save(checkpoint, path)


def test_resume_default_none_and_explicit_exact_preserve_exact_semantics(
    tmp_path: Path,
) -> None:
    case = PREFIX + "DeterministicProcessCase"
    for index, comparison in enumerate(("default", None, ExactComparison())):
        kwargs: dict[str, object] = {"temporary_root": tmp_path / str(index)}
        if comparison != "default":
            kwargs["comparison"] = comparison
        result = check_resume(case, **kwargs)  # type: ignore[arg-type]
        assert result.outcome is Outcome.PASS
        assert result.comparison_policy == "exact"
        assert result.comparison_rtol is None
        assert result.comparison_atol is None
        assert result.comparison_equal_nan is None

    exact = ProcessResumeRunner(
        comparison=ExactComparison(), temporary_root=tmp_path / "exact"
    ).run(case, staged_checkpoint_hook=_small_staged_model_nudge)
    assert exact.outcome is Outcome.FAIL

    for index, comparison in enumerate(("default", None, ExactComparison())):
        kwargs: dict[str, object] = {"temporary_root": tmp_path / f"fault-{index}"}
        if comparison != "default":
            kwargs["comparison"] = comparison
        fault = check_resume("trainparity.quickstarts.resume:FaultyCase", **kwargs)  # type: ignore[arg-type]
        abstain = check_resume(PREFIX + "NondeterministicProcessCase", **kwargs)  # type: ignore[arg-type]
        assert fault.outcome is Outcome.FAIL
        assert abstain.outcome is Outcome.ABSTAIN


def test_resume_explicit_tolerance_controls_candidate_and_report(tmp_path: Path) -> None:
    result = ProcessResumeRunner(
        comparison=ToleranceComparison(rtol=1e-5, atol=1e-8),
        temporary_root=tmp_path,
    ).run(PREFIX + "DeterministicProcessCase", staged_checkpoint_hook=_small_staged_model_nudge)
    assert result.outcome is Outcome.PASS
    assert "declared tolerance" in result.message
    assert result.comparison_policy == "explicit_tolerance"
    assert result.comparison_rtol == 1e-5
    assert result.comparison_atol == 1e-8
    assert result.comparison_equal_nan is False
    assert result.to_dict()["comparison_policy"] == "explicit_tolerance"
    json.dumps(result.to_dict(), sort_keys=True)


def test_resume_tolerance_fail_reports_numeric_error(tmp_path: Path) -> None:
    result = ProcessResumeRunner(
        comparison=ToleranceComparison(rtol=0.0, atol=1e-9),
        temporary_root=tmp_path,
    ).run(PREFIX + "DeterministicProcessCase", staged_checkpoint_hook=_small_staged_model_nudge)
    assert result.outcome is Outcome.FAIL
    assert result.primary_difference is not None
    assert result.primary_difference.max_abs_error is not None
    assert result.primary_difference.max_rel_error is not None


def test_resume_uses_the_same_policy_for_baseline_self_consistency(tmp_path: Path) -> None:
    exact = ProcessResumeRunner(temporary_root=tmp_path / "exact").run(
        PREFIX + "BaselineToleranceProcessCase"
    )
    tolerant = ProcessResumeRunner(
        comparison=ToleranceComparison(rtol=1e-5, atol=1e-8),
        temporary_root=tmp_path / "tolerant",
    ).run(PREFIX + "BaselineToleranceProcessCase")
    outside = ProcessResumeRunner(
        comparison=ToleranceComparison(rtol=1e-5, atol=1e-8),
        temporary_root=tmp_path / "outside",
    ).run(PREFIX + "BaselineOutsideToleranceProcessCase")
    assert exact.outcome is Outcome.ABSTAIN
    assert tolerant.outcome is Outcome.PASS
    assert outside.outcome is Outcome.ABSTAIN


@pytest.mark.parametrize("comparison", ("tolerance", {}, lambda: None, 1.0))
def test_resume_rejects_non_policy_comparisons(comparison: object) -> None:
    with pytest.raises(TypeError, match="comparison must be ExactComparison or ToleranceComparison"):
        check_resume(PREFIX + "DeterministicProcessCase", comparison=comparison)  # type: ignore[arg-type]


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


def test_post_run_checkpoint_lookup_exception_is_error(tmp_path: Path) -> None:
    result = ProcessResumeRunner(temporary_root=tmp_path).run(
        PREFIX + "PostRunCheckpointPathErrorCase"
    )
    assert result.outcome is Outcome.ERROR
    assert result.processes[0].phase == "baseline_a"
    assert result.processes[0].returncode == 0
    assert result.message == (
        "baseline_a checkpoint path resolution after child completion failed: "
        "FileNotFoundError"
    )


def test_candidate_resume_staging_path_exception_is_error(tmp_path: Path) -> None:
    result = ProcessResumeRunner(temporary_root=tmp_path).run(
        PREFIX + "CandidateResumeCheckpointPathErrorCase"
    )
    assert result.outcome is Outcome.ERROR
    assert [process.phase for process in result.processes] == [
        "baseline_a",
        "baseline_b",
        "candidate_split",
    ]
    assert result.message == (
        "candidate_resume checkpoint path resolution before child launch failed: "
        "FileNotFoundError"
    )


def test_command_and_observation_callback_exceptions_are_errors(tmp_path: Path) -> None:
    command = ProcessResumeRunner(temporary_root=tmp_path / "command").run(
        PREFIX + "CommandCallbackErrorCase"
    )
    observation = ProcessResumeRunner(temporary_root=tmp_path / "observation").run(
        PREFIX + "ObservationCallbackErrorCase"
    )
    assert command.outcome is Outcome.ERROR
    assert command.message == "baseline_a command callback or validation failed: RuntimeError"
    assert observation.outcome is Outcome.ERROR
    assert observation.message == (
        "baseline_a checkpoint observation failed: "
        "observe_checkpoint callback failed: RuntimeError"
    )


def test_callback_base_exception_is_not_converted(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="7"):
        ProcessResumeRunner(temporary_root=tmp_path).run(PREFIX + "SystemExitCommandCase")


def test_staging_copy_failure_remains_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_copy(source: Path, destination: Path) -> None:
        raise OSError("intentional copy failure")

    monkeypatch.setattr("trainparity.process_resume.shutil.copy2", fail_copy)
    result = ProcessResumeRunner(temporary_root=tmp_path).run(
        PREFIX + "DeterministicProcessCase"
    )
    assert result.outcome is Outcome.ERROR
    assert result.message == "candidate_resume checkpoint staging failed: OSError"


def test_preserved_work_dir_makes_child_stderr_discoverable(tmp_path: Path) -> None:
    work_dir = tmp_path / "preserved"
    result = ProcessResumeRunner().run(PREFIX + "ErrorProcessCase", work_dir=work_dir)
    assert result.outcome is Outcome.ERROR
    assert "baseline_a/stderr.log under the preserved work_dir" in result.message
    stderr = work_dir / "baseline_a" / "stderr.log"
    assert stderr.is_file()
    assert "intentional child failure" in stderr.read_text(encoding="utf-8")
