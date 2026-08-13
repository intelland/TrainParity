from __future__ import annotations

import math
import struct
from dataclasses import replace

import pytest
import torch

from trainparity import ExactComparison, Outcome, ToleranceComparison
from trainparity.snapshot import Snapshot
from trainparity.state import FrozenMapping, FrozenTensor, FullValueBackend


def snapshot(value: object, *, step: int | None = 0) -> Snapshot:
    frozen = FullValueBackend().freeze(value)
    assert isinstance(frozen, FrozenMapping)
    return Snapshot(step, frozen)


def difference(left: object, right: object, *, tolerance: bool = False):
    policy = ToleranceComparison(rtol=1e-4, atol=1e-6) if tolerance else ExactComparison()
    result = policy.compare(snapshot({"extra": left}), snapshot({"extra": right}))
    assert result.outcome is Outcome.FAIL
    assert result.first_difference is not None
    assert "first observed divergence" in result.message
    assert "root cause" not in result.message
    return result.first_difference


def test_clean_nested_dict_list_tuple_has_no_false_positive() -> None:
    state = {"nested": {"items": [1, None, ("x", 0.0)]}}
    result = ExactComparison().compare(snapshot(state), snapshot(state))
    assert result.outcome is Outcome.PASS
    assert result.first_difference is None


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        ({"a": 1}, {}, "missing_candidate"),
        ({}, {"a": 1}, "missing_baseline"),
        (None, 0, "type"),
        ([], [0], "sequence_length"),
        ([], (), "sequence_kind"),
        (True, 1, "type"),
    ],
)
def test_missing_none_empty_zero_and_structure_are_distinct(
    left: object, right: object, reason: str
) -> None:
    observed = difference(left, right)
    assert observed.reason == reason
    assert observed.path.startswith("extra")
    assert observed.baseline != observed.candidate


def test_same_value_at_different_paths_reports_stable_first_path() -> None:
    observed = difference({"a": 1}, {"b": 1})
    assert observed.path == "extra.a"
    assert observed.reason == "missing_candidate"


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (torch.ones(2), torch.ones(3), "tensor_shape"),
        (torch.ones(2), torch.ones(2, dtype=torch.float64), "tensor_dtype"),
        (torch.tensor([1.0]), torch.tensor([2.0]), "tensor_value"),
    ],
)
def test_tensor_shape_dtype_and_value_are_separate(
    left: torch.Tensor, right: torch.Tensor, reason: str
) -> None:
    observed = difference(left, right)
    assert observed.path == "extra"
    assert observed.reason == reason


def test_tensor_device_and_requires_grad_metadata_are_exact() -> None:
    frozen = FrozenTensor.capture(torch.tensor([1.0]))
    left = Snapshot(0, FrozenMapping((("value", frozen),)))
    fake_cuda = replace(frozen, device="cuda:0")
    observed = ExactComparison().compare(left, Snapshot(0, FrozenMapping((("value", fake_cuda),))))
    assert observed.first_difference is not None
    assert observed.first_difference.reason == "tensor_device"
    requires_grad = replace(frozen, requires_grad=True)
    observed = ExactComparison().compare(
        left, Snapshot(0, FrozenMapping((("value", requires_grad),)))
    )
    assert observed.first_difference is not None
    assert observed.first_difference.reason == "tensor_requires_grad"


def test_exact_distinguishes_signed_zero_and_tolerance_does_not() -> None:
    assert difference(0.0, -0.0).reason == "value"
    result = ToleranceComparison(rtol=0.0, atol=0.0).compare(
        snapshot({"value": 0.0}), snapshot({"value": -0.0})
    )
    assert result.outcome is Outcome.PASS


def test_nan_and_inf_policies_are_explicit() -> None:
    exact_nan = ExactComparison().compare(snapshot({"x": math.nan}), snapshot({"x": math.nan}))
    assert exact_nan.outcome is Outcome.PASS
    unequal_nan = struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    assert difference(math.nan, unequal_nan).reason == "value"
    assert difference(math.inf, -math.inf).reason == "value"
    assert ToleranceComparison(rtol=0, atol=0, equal_nan=True).compare(
        snapshot({"x": math.nan}), snapshot({"x": math.nan})
    ).outcome is Outcome.PASS
    assert ToleranceComparison(rtol=0, atol=0, equal_nan=False).compare(
        snapshot({"x": math.nan}), snapshot({"x": math.nan})
    ).outcome is Outcome.FAIL


def test_tensor_nan_inf_empty_and_tolerance() -> None:
    left = snapshot({"x": torch.tensor([math.nan, math.inf, 1.0]), "empty": torch.empty(0)})
    right = snapshot({"x": torch.tensor([math.nan, math.inf, 1.00001]), "empty": torch.empty(0)})
    assert ExactComparison().compare(left, right).outcome is Outcome.FAIL
    strict = ToleranceComparison(rtol=0, atol=0, equal_nan=False).compare(left, right)
    assert strict.outcome is Outcome.FAIL
    tolerant = ToleranceComparison(rtol=1e-4, atol=0, equal_nan=True).compare(left, right)
    assert tolerant.outcome is Outcome.PASS


def test_tolerance_reports_numerical_error() -> None:
    observed = difference(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]), tolerance=True)
    assert observed.max_abs_error == 1.0
    assert observed.max_rel_error == 0.5


def test_exact_float_tensor_mismatch_reports_error_without_weakening_exactness() -> None:
    result = ExactComparison().compare(
        snapshot({"x": torch.tensor([1.0, 2.0])}),
        snapshot({"x": torch.tensor([1.0, 2.000001])}),
    )
    assert result.outcome is Outcome.FAIL
    assert result.first_difference is not None
    assert result.first_difference.path == "x"
    assert result.first_difference.reason == "tensor_value"
    assert result.first_difference.max_abs_error is not None
    assert result.first_difference.max_rel_error is not None


def test_integral_tensor_tolerance_remains_exact() -> None:
    observed = difference(torch.tensor([1]), torch.tensor([2]), tolerance=True)
    assert observed.reason == "tensor_value"


@pytest.mark.parametrize("rtol,atol", [(-1.0, 0.0), (0.0, -1.0), (math.inf, 0), (math.nan, 0)])
def test_invalid_tolerances_are_rejected(rtol: float, atol: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        ToleranceComparison(rtol=rtol, atol=atol)


def test_snapshot_metadata_is_compared_before_state() -> None:
    base = snapshot({"x": 1})
    assert ExactComparison().compare(base, replace(base, schema_version=2)).first_difference.path == "schema_version"  # type: ignore[union-attr]
    assert ExactComparison().compare(base, replace(base, backend="other")).first_difference.path == "backend"  # type: ignore[union-attr]
    assert ExactComparison().compare(base, replace(base, step=2)).first_difference.path == "step"  # type: ignore[union-attr]
