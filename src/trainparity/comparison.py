"""Separate exact and explicit-tolerance snapshot comparison policies."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import torch

from trainparity.outcomes import Outcome
from trainparity.snapshot import Snapshot
from trainparity.state import (
    FrozenMapping,
    FrozenSequence,
    FrozenTensor,
    FrozenValue,
    StatePath,
    render_path,
)


@dataclass(frozen=True)
class Difference:
    """A stable first-observed state difference, never a root-cause claim."""

    path: str
    reason: str
    baseline: str
    candidate: str
    max_abs_error: float | None = None
    max_rel_error: float | None = None


@dataclass(frozen=True)
class ComparisonResult:
    """A non-Boolean comparison outcome with actionable difference detail."""

    outcome: Outcome
    first_difference: Difference | None
    message: str


class ExactComparison:
    """Bitwise tensor/float policy with exact metadata and structure."""

    def compare(self, baseline: Snapshot, candidate: Snapshot) -> ComparisonResult:
        """Return the first deterministic difference between two snapshots."""
        difference = _snapshot_difference(baseline, candidate, tolerance=None)
        return _result(difference)

    def compare_all(self, baseline: Snapshot, candidate: Snapshot) -> tuple[Difference, ...]:
        """Return every deterministically ordered difference in this snapshot pair."""
        return _snapshot_differences(baseline, candidate, tolerance=None)


class ToleranceComparison:
    """Explicit numerical policy; structure and tensor metadata stay exact."""

    def __init__(self, *, rtol: float, atol: float, equal_nan: bool = False) -> None:
        if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
            raise ValueError("rtol and atol must be finite and non-negative")
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan

    def compare(self, baseline: Snapshot, candidate: Snapshot) -> ComparisonResult:
        """Return the first difference outside the explicit tolerance."""
        difference = _snapshot_difference(
            baseline,
            candidate,
            tolerance=(self.rtol, self.atol, self.equal_nan),
        )
        return _result(difference)

    def compare_all(self, baseline: Snapshot, candidate: Snapshot) -> tuple[Difference, ...]:
        """Return every difference outside the explicit tolerance."""
        return _snapshot_differences(
            baseline,
            candidate,
            tolerance=(self.rtol, self.atol, self.equal_nan),
        )


def _result(difference: Difference | None) -> ComparisonResult:
    if difference is None:
        return ComparisonResult(Outcome.PASS, None, "snapshots are equivalent under policy")
    return ComparisonResult(
        Outcome.FAIL,
        difference,
        f"first observed divergence at {difference.path}: {difference.reason}",
    )


def _snapshot_difference(
    baseline: Snapshot,
    candidate: Snapshot,
    tolerance: tuple[float, float, bool] | None,
) -> Difference | None:
    if baseline.schema_version != candidate.schema_version:
        return _difference(
            ("schema_version",),
            "schema_version",
            baseline.schema_version,
            candidate.schema_version,
        )
    if baseline.backend != candidate.backend:
        return _difference(("backend",), "backend", baseline.backend, candidate.backend)
    if baseline.step != candidate.step:
        return _difference(("step",), "step", baseline.step, candidate.step)
    return _value_difference(baseline.state, candidate.state, (), tolerance)


def _snapshot_differences(
    baseline: Snapshot,
    candidate: Snapshot,
    tolerance: tuple[float, float, bool] | None,
) -> tuple[Difference, ...]:
    differences: list[Difference] = []
    if baseline.schema_version != candidate.schema_version:
        differences.append(
            _difference(
                ("schema_version",),
                "schema_version",
                baseline.schema_version,
                candidate.schema_version,
            )
        )
    if baseline.backend != candidate.backend:
        differences.append(_difference(("backend",), "backend", baseline.backend, candidate.backend))
    if baseline.step != candidate.step:
        differences.append(_difference(("step",), "step", baseline.step, candidate.step))
    differences.extend(_all_value_differences(baseline.state, candidate.state, (), tolerance))
    return tuple(differences)


class _Missing:
    pass


_MISSING = _Missing()


def _value_difference(
    baseline: FrozenValue | _Missing,
    candidate: FrozenValue | _Missing,
    path: StatePath,
    tolerance: tuple[float, float, bool] | None,
) -> Difference | None:
    if isinstance(baseline, _Missing) or isinstance(candidate, _Missing):
        reason = "missing_candidate" if isinstance(candidate, _Missing) else "missing_baseline"
        return _difference(path, reason, baseline, candidate)
    if type(baseline) is not type(candidate):
        return _difference(path, "type", baseline, candidate)
    if isinstance(baseline, FrozenMapping) and isinstance(candidate, FrozenMapping):
        return _mapping_difference(baseline, candidate, path, tolerance)
    if isinstance(baseline, FrozenSequence) and isinstance(candidate, FrozenSequence):
        if baseline.kind != candidate.kind:
            return _difference(path, "sequence_kind", baseline.kind, candidate.kind)
        if len(baseline.items) != len(candidate.items):
            return _difference(path, "sequence_length", len(baseline.items), len(candidate.items))
        for index, (left, right) in enumerate(zip(baseline.items, candidate.items, strict=True)):
            difference = _value_difference(left, right, (*path, index), tolerance)
            if difference is not None:
                return difference
        return None
    if isinstance(baseline, FrozenTensor) and isinstance(candidate, FrozenTensor):
        return _tensor_difference(baseline, candidate, path, tolerance)
    if isinstance(baseline, float) and isinstance(candidate, float):
        return _float_difference(baseline, candidate, path, tolerance)
    if baseline != candidate:
        return _difference(path, "value", baseline, candidate)
    return None


def _mapping_difference(
    baseline: FrozenMapping,
    candidate: FrozenMapping,
    path: StatePath,
    tolerance: tuple[float, float, bool] | None,
) -> Difference | None:
    left = dict(baseline.entries)
    right = dict(candidate.entries)
    for key in sorted(left.keys() | right.keys()):
        difference = _value_difference(
            left.get(key, _MISSING),
            right.get(key, _MISSING),
            (*path, key),
            tolerance,
        )
        if difference is not None:
            return difference
    return None


def _all_value_differences(
    baseline: FrozenValue | _Missing,
    candidate: FrozenValue | _Missing,
    path: StatePath,
    tolerance: tuple[float, float, bool] | None,
) -> list[Difference]:
    if isinstance(baseline, _Missing) or isinstance(candidate, _Missing):
        reason = "missing_candidate" if isinstance(candidate, _Missing) else "missing_baseline"
        return [_difference(path, reason, baseline, candidate)]
    if type(baseline) is not type(candidate):
        return [_difference(path, "type", baseline, candidate)]
    if isinstance(baseline, FrozenMapping) and isinstance(candidate, FrozenMapping):
        left = dict(baseline.entries)
        right = dict(candidate.entries)
        differences: list[Difference] = []
        for key in sorted(left.keys() | right.keys()):
            differences.extend(
                _all_value_differences(
                    left.get(key, _MISSING),
                    right.get(key, _MISSING),
                    (*path, key),
                    tolerance,
                )
            )
        return differences
    if isinstance(baseline, FrozenSequence) and isinstance(candidate, FrozenSequence):
        if baseline.kind != candidate.kind:
            return [_difference(path, "sequence_kind", baseline.kind, candidate.kind)]
        if len(baseline.items) != len(candidate.items):
            return [_difference(path, "sequence_length", len(baseline.items), len(candidate.items))]
        differences = []
        for index, (sequence_left, sequence_right) in enumerate(
            zip(baseline.items, candidate.items, strict=True)
        ):
            differences.extend(
                _all_value_differences(sequence_left, sequence_right, (*path, index), tolerance)
            )
        return differences
    difference = _value_difference(baseline, candidate, path, tolerance)
    return [] if difference is None else [difference]


def _tensor_difference(
    baseline: FrozenTensor,
    candidate: FrozenTensor,
    path: StatePath,
    tolerance: tuple[float, float, bool] | None,
) -> Difference | None:
    for field in ("shape", "dtype", "device", "requires_grad"):
        left = getattr(baseline, field)
        right = getattr(candidate, field)
        if left != right:
            return _difference(path, f"tensor_{field}", left, right)
    if tolerance is None:
        if baseline.data != candidate.data:
            if baseline.to_tensor().is_floating_point() or baseline.to_tensor().is_complex():
                max_abs, max_rel = _errors(baseline.to_tensor(), candidate.to_tensor())
                return _difference(
                    path, "tensor_value", baseline, candidate, max_abs, max_rel
                )
            return _difference(path, "tensor_value", baseline, candidate)
        return None
    left_tensor = baseline.to_tensor()
    right_tensor = candidate.to_tensor()
    rtol, atol, equal_nan = tolerance
    if left_tensor.is_floating_point() or left_tensor.is_complex():
        close = torch.isclose(left_tensor, right_tensor, rtol=rtol, atol=atol, equal_nan=equal_nan)
    else:
        close = torch.eq(left_tensor, right_tensor)
    if bool(torch.all(close)):
        return None
    max_abs, max_rel = _errors(left_tensor, right_tensor)
    return _difference(path, "tensor_value", baseline, candidate, max_abs, max_rel)


def _float_difference(
    baseline: float,
    candidate: float,
    path: StatePath,
    tolerance: tuple[float, float, bool] | None,
) -> Difference | None:
    if tolerance is None:
        equal = struct.pack(">d", baseline) == struct.pack(">d", candidate)
    else:
        rtol, atol, equal_nan = tolerance
        if math.isnan(baseline) or math.isnan(candidate):
            equal = equal_nan and math.isnan(baseline) and math.isnan(candidate)
        elif math.isinf(baseline) or math.isinf(candidate):
            equal = baseline == candidate
        else:
            equal = math.isclose(baseline, candidate, rel_tol=rtol, abs_tol=atol)
    return None if equal else _difference(path, "value", baseline, candidate)


def _errors(baseline: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    if baseline.numel() == 0:
        return 0.0, 0.0
    left = baseline.to(torch.complex128 if baseline.is_complex() else torch.float64)
    right = candidate.to(torch.complex128 if candidate.is_complex() else torch.float64)
    absolute = torch.abs(left - right)
    finite_absolute = absolute[torch.isfinite(absolute)]
    max_abs = float(finite_absolute.max()) if finite_absolute.numel() else math.inf
    denominator = torch.abs(left)
    relative = torch.where(denominator > 0, absolute / denominator, absolute)
    finite_relative = relative[torch.isfinite(relative)]
    max_rel = float(finite_relative.max()) if finite_relative.numel() else math.inf
    return max_abs, max_rel


def _difference(
    path: StatePath,
    reason: str,
    baseline: object,
    candidate: object,
    max_abs_error: float | None = None,
    max_rel_error: float | None = None,
) -> Difference:
    return Difference(
        render_path(path),
        reason,
        _summary(baseline),
        _summary(candidate),
        max_abs_error,
        max_rel_error,
    )


def _summary(value: object) -> str:
    if isinstance(value, _Missing):
        return "<missing>"
    if isinstance(value, FrozenTensor):
        tensor = value.to_tensor()
        sample = tensor.flatten()[:4].tolist()
        return (
            f"tensor(shape={value.shape}, dtype={value.dtype}, device={value.device}, "
            f"values={sample})"
        )
    if isinstance(value, FrozenMapping):
        return f"mapping(keys={[key for key, _ in value.entries]})"
    if isinstance(value, FrozenSequence):
        return f"{value.kind}(length={len(value.items)})"
    return repr(value)
