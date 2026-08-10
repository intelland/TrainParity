"""Run Gate 2 snapshot/comparator contract cases and record stable paths."""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from trainparity import ExactComparison, Outcome, Snapshot, ToleranceComparison, capture_snapshot
from trainparity.state import FrozenMapping, FrozenTensor, FullValueBackend


def snapshot(value: dict[str, object]) -> Snapshot:
    """Build a reference-backend snapshot for a tiny contract case."""
    frozen = FullValueBackend().freeze(value)
    assert isinstance(frozen, FrozenMapping)
    return Snapshot(0, frozen)


def captured(model: nn.Module, optimizer: torch.optim.Optimizer | None = None, **kwargs: Any) -> Snapshot:
    """Require successful capture for a fault-suite setup."""
    result = capture_snapshot(model, optimizer=optimizer, capture_rng=False, **kwargs)
    if result.outcome is not Outcome.PASS or result.snapshot is None:
        raise RuntimeError(f"fixture capture failed: {result}")
    return result.snapshot


def one_parameter_model() -> nn.Module:
    """Return a model with one stable parameter name."""

    class OneParameter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0]))

    return OneParameter()


def optimizer_state_pair(kind: str) -> tuple[Snapshot, Snapshot]:
    """Capture optimizer state before and after a named state mutation."""
    model = one_parameter_model()
    optimizer: torch.optim.Optimizer
    if kind == "momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sum(model.parameters()).sum().backward()
    optimizer.step()
    baseline = captured(model, optimizer)
    state = next(iter(optimizer.state.values()))
    key = "momentum_buffer" if kind == "momentum" else "exp_avg"
    state[key].add_(1)
    return baseline, captured(model, optimizer)


def parameter_group_pair() -> tuple[Snapshot, Snapshot]:
    """Create snapshots whose first optimizer group learning rate differs."""
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
    left = torch.optim.SGD(
        [{"params": model[0].parameters(), "lr": 0.1}, {"params": model[1].parameters(), "lr": 0.2}]
    )
    right = torch.optim.SGD(
        [{"params": model[0].parameters(), "lr": 0.2}, {"params": model[1].parameters(), "lr": 0.1}]
    )
    return captured(model, left), captured(model, right)


def extra_state_pair() -> tuple[Snapshot, Snapshot]:
    """Create a difference through two user-provided state_dict objects."""

    class Extra:
        def __init__(self, value: int) -> None:
            self.value = value

        def state_dict(self) -> dict[str, object]:
            return {"shadow": {"weight": self.value}}

    model = one_parameter_model()
    return (
        captured(model, extras={"ema": Extra(1)}),
        captured(model, extras={"ema": Extra(2)}),
    )


def device_pair() -> tuple[Snapshot, Snapshot]:
    """Exercise device metadata without requiring a CUDA allocation."""
    frozen = FrozenTensor.capture(torch.tensor([1.0]))
    return (
        Snapshot(0, FrozenMapping((("extra", frozen),))),
        Snapshot(0, FrozenMapping((("extra", replace(frozen, device="cuda:0")),))),
    )


def evaluate() -> dict[str, Any]:
    """Run every required clean/fault contract case."""
    momentum = optimizer_state_pair("momentum")
    adam = optimizer_state_pair("adam")
    groups = parameter_group_pair()
    extras = extra_state_pair()
    devices = device_pair()
    cases: dict[str, tuple[Snapshot, Snapshot, str]] = {
        "nested_value": (
            snapshot({"extra": {"nested": {"items": [{"value": 1}]}}}),
            snapshot({"extra": {"nested": {"items": [{"value": 2}]}}}),
            "extra.nested.items[0].value",
        ),
        "tensor_shape": (
            snapshot({"extra": {"tensor": torch.ones(2)}}),
            snapshot({"extra": {"tensor": torch.ones(3)}}),
            "extra.tensor",
        ),
        "tensor_dtype": (
            snapshot({"extra": {"tensor": torch.ones(2)}}),
            snapshot({"extra": {"tensor": torch.ones(2, dtype=torch.float64)}}),
            "extra.tensor",
        ),
        "tensor_value": (
            snapshot({"extra": {"tensor": torch.tensor([1.0])}}),
            snapshot({"extra": {"tensor": torch.tensor([2.0])}}),
            "extra.tensor",
        ),
        "nan": (snapshot({"extra": math.nan}), snapshot({"extra": 0.0}), "extra"),
        "inf": (snapshot({"extra": math.inf}), snapshot({"extra": -math.inf}), "extra"),
        "empty_tensor": (
            snapshot({"extra": torch.empty(0)}),
            snapshot({"extra": torch.ones(1)}),
            "extra",
        ),
        "none_vs_zero": (snapshot({"extra": None}), snapshot({"extra": 0}), "extra"),
        "missing_vs_none": (
            snapshot({"extra": {}}),
            snapshot({"extra": {"value": None}}),
            "extra.value",
        ),
        "device_metadata": (*devices, "extra"),
        "parameter_group_order": (*groups, "optimizer.param_groups[0].lr"),
        "sgd_momentum": (*momentum, "optimizer.state.weight.momentum_buffer"),
        "adam_exp_avg": (*adam, "optimizer.state.weight.exp_avg"),
        "missing_key": (
            snapshot({"extra": {"a": 1}}),
            snapshot({"extra": {}}),
            "extra.a",
        ),
        "extra_key": (
            snapshot({"extra": {}}),
            snapshot({"extra": {"b": 1}}),
            "extra.b",
        ),
        "same_value_different_path": (
            snapshot({"extra": {"a": 1}}),
            snapshot({"extra": {"b": 1}}),
            "extra.a",
        ),
        "user_state_dict": (*extras, "extra.ema.shadow.weight"),
    }
    exact = ExactComparison()
    fault_results = []
    clean_results = []
    for name, (baseline, candidate, expected_path) in cases.items():
        result = exact.compare(baseline, candidate)
        difference = asdict(result.first_difference) if result.first_difference else None
        fault_results.append(
            {
                "case": name,
                "outcome": result.outcome.value,
                "expected_path": expected_path,
                "observed_path": result.first_difference.path if result.first_difference else None,
                "difference": difference,
            }
        )
        clean = exact.compare(baseline, baseline)
        clean_results.append({"case": name, "outcome": clean.outcome.value})

    small_left = snapshot({"extra": torch.tensor([1.0])})
    small_right = snapshot({"extra": torch.tensor([1.00001])})
    exact_small = exact.compare(small_left, small_right)
    tolerance_small = ToleranceComparison(rtol=1e-4, atol=0, equal_nan=False).compare(
        small_left, small_right
    )

    class Alias(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.alias = self.weight

    alias = Alias()
    ambiguous = capture_snapshot(
        alias,
        optimizer=torch.optim.SGD(alias.parameters(), lr=0.1),
        capture_rng=False,
    )
    mutable = torch.tensor([3.0])
    frozen_alias = snapshot({"extra": mutable})
    mutable.add_(9)
    alias_reference = snapshot({"extra": torch.tensor([3.0])})
    alias_result = exact.compare(frozen_alias, alias_reference)
    return {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "faults": fault_results,
        "clean": clean_results,
        "fault_count": len(fault_results),
        "faults_with_expected_path": sum(
            item["observed_path"] == item["expected_path"] for item in fault_results
        ),
        "clean_false_positives": sum(item["outcome"] != "PASS" for item in clean_results),
        "policy_separation": {
            "exact": exact_small.outcome.value,
            "tolerance": tolerance_small.outcome.value,
        },
        "ambiguous_optimizer": {
            "outcome": ambiguous.outcome.value,
            "path": ambiguous.issue.path if ambiguous.issue else None,
            "detail": ambiguous.issue.detail if ambiguous.issue else None,
        },
        "tensor_alias_frozen": alias_result.outcome.value,
        "scope": "Gate 2 snapshot/comparison only; no resume orchestration",
    }


def main() -> None:
    """Write the Gate 2 evidence JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "faults": report["fault_count"],
                "matched": report["faults_with_expected_path"],
                "clean_false_positives": report["clean_false_positives"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
