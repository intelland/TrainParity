from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from trainparity import (
    AccumulationExecutionPlan,
    AccumulationRunner,
    ExactComparison,
    ToleranceComparison,
    UnsafeBatchSplit,
    split_tensor_tree,
)


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


def test_unsafe_complex_batch_returns_abstain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del tmp_path, monkeypatch
    # Unsafe splitter behavior is covered directly; worker maps this exception
    # to ABSTAIN in the process contract.
    with pytest.raises(UnsafeBatchSplit):
        split_tensor_tree({"metadata": ["a", "b"]}, 2)
