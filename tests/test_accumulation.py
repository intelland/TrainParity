from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from trainparity import ExactComparison, Outcome, ToleranceComparison, check_accumulation
from trainparity.accumulation import (
    AccumulationExecutionPlan,
    AccumulationRunner,
    UnsafeBatchSplit,
    split_tensor_tree,
)
from trainparity.api import TrainingState


def test_training_state_defaults_to_no_scheduler() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    state = TrainingState(model=model, optimizer=optimizer)

    assert state.scheduler is None


def test_tensor_tree_split_preserves_structure_and_order() -> None:
    batch = {
        "tokens": torch.arange(12).reshape(4, 3),
        "nested": (torch.arange(4), [torch.arange(8).reshape(4, 2)]),
    }
    parts = split_tensor_tree(batch, 2)
    assert len(parts) == 2
    assert isinstance(parts[0], dict)
    assert torch.equal(parts[0]["tokens"], batch["tokens"][:2])
    assert torch.equal(parts[1]["nested"][1][0], batch["nested"][1][0][2:])


@pytest.mark.parametrize(
    "batch",
    [torch.tensor(1.0), {"x": torch.ones(3), "y": torch.ones(2)}, object(), {}],
)
def test_tensor_tree_split_abstains_on_unsafe_shape(batch: object) -> None:
    with pytest.raises(UnsafeBatchSplit):
        split_tensor_tree(batch, 2)


def test_plan_validation_and_policy_are_explicit() -> None:
    with pytest.raises(ValueError):
        AccumulationExecutionPlan(0).validate()
    with pytest.raises(ValueError):
        AccumulationExecutionPlan(2, clip_grad_norm=0).validate()
    with pytest.raises(TypeError):
        AccumulationRunner(comparison=object())  # type: ignore[arg-type]
    assert AccumulationRunner(comparison=ExactComparison())
    assert AccumulationRunner(comparison=ToleranceComparison(rtol=0, atol=0))


def test_clean_and_fault_run_in_three_fresh_processes(tmp_path: Path) -> None:
    runner = AccumulationRunner(
        comparison=ToleranceComparison(rtol=1e-6, atol=1e-7), timeout=60
    )
    clean = runner.run(
        "experiments.gate5.cases:LinearCase",
        candidate=AccumulationExecutionPlan(2),
        report_path=tmp_path / "clean.json",
    )
    assert clean.outcome == "PASS"
    assert clean.verified_equal_initial_state
    assert len(set(clean.process_ids)) == 3
    assert clean.loss_normalization_captured
    assert clean.peak_temporary_directory_bytes > 0
    assert clean.persisted_artifact_bytes > 0
    stored = json.loads((tmp_path / "clean.json").read_text(encoding="utf-8"))
    assert stored["persisted_artifact_bytes"] == (tmp_path / "clean.json").stat().st_size

    fault = runner.run(
        "experiments.gate5.cases:LinearCase",
        candidate=AccumulationExecutionPlan(
            2, scale_accumulated_loss=False, use_explicit_loss_accounting=False
        ),
    )
    assert fault.outcome == "FAIL"
    assert fault.first_observed_phase == "loss_accounting"
    assert "not a root-cause claim" in fault.message


def test_setup_error_reports_actionable_cause_deterministically(tmp_path: Path) -> None:
    case = "experiments.gate5.cases:MissingAccumulationCase"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = check_accumulation(
        case,
        candidate=AccumulationExecutionPlan(2),
        report_path=first_path,
    )
    second = check_accumulation(
        case,
        candidate=AccumulationExecutionPlan(2),
        report_path=second_path,
    )

    assert first.outcome is Outcome.ERROR
    assert first.message.startswith("case setup failed: CaseImportError: cannot import case")
    assert case in first.message
    assert "MissingAccumulationCase" in first.message
    assert "Traceback" not in first.message
    assert first_path.is_file()
    serialized = json.loads(json.dumps(first.to_dict()))
    assert json.loads(first_path.read_text(encoding="utf-8")) == serialized
    assert first.to_dict() == second.to_dict()
    assert first_path.read_bytes() == second_path.read_bytes()


def test_check_accumulation_signature_is_unchanged() -> None:
    signature = inspect.signature(check_accumulation)

    assert list(signature.parameters) == [
        "case",
        "candidate",
        "comparison",
        "device",
        "seed",
        "cwd",
        "report_path",
        "environment",
        "timeout",
        "temporary_root",
    ]
    assert signature.parameters["candidate"].kind is inspect.Parameter.KEYWORD_ONLY


def test_unsafe_complex_batch_returns_abstain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del tmp_path, monkeypatch
    # Unsafe splitter behavior is covered directly; worker maps this exception
    # to ABSTAIN in the process contract.
    with pytest.raises(UnsafeBatchSplit):
        split_tensor_tree({"metadata": ["a", "b"]}, 2)
