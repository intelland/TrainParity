"""Run clean and intentionally faulty CPU accumulation checks."""

from __future__ import annotations

import json

import torch
from torch import nn

from trainparity.api import (
    MACHINE_REPORT_SCHEMA_VERSION,
    AccumulationExecutionPlan,
    LossAccounting,
    ToleranceComparison,
    TrainingState,
    check_accumulation,
)
from trainparity.version import PACKAGE_VERSION


class Case:
    """A declared global-mean equivalence over one optimizer update."""

    equivalence = "same four ordered samples; global squared-error mean"

    def build(self, seed: int, device: str) -> TrainingState:
        torch.manual_seed(seed)
        model = nn.Linear(1, 1, bias=False).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        return TrainingState(model, optimizer, scheduler)

    def batch(self, device: str) -> object:
        values = torch.arange(1, 5, dtype=torch.float32, device=device).reshape(-1, 1)
        return values, values * 0.5

    def loss(self, state: TrainingState, batch: object) -> LossAccounting:
        assert isinstance(batch, tuple)
        inputs, targets = batch
        errors = (state.model(inputs) - targets).square()
        numerator = errors.sum()
        denominator = int(errors.numel())
        return LossAccounting(numerator / denominator, numerator, denominator)


def run() -> dict[str, object]:
    """Return a clean PASS and a missing-loss-scaling FAIL."""
    comparison = ToleranceComparison(rtol=1e-6, atol=1e-7)
    clean = check_accumulation(
        "trainparity.quickstarts.accumulation:Case",
        candidate=AccumulationExecutionPlan(2),
        comparison=comparison,
    )
    fault = check_accumulation(
        "trainparity.quickstarts.accumulation:Case",
        candidate=AccumulationExecutionPlan(
            2,
            scale_accumulated_loss=False,
            use_explicit_loss_accounting=False,
        ),
        comparison=comparison,
    )
    return {
        "schema_version": MACHINE_REPORT_SCHEMA_VERSION,
        "trainparity_version": PACKAGE_VERSION,
        "clean": clean.to_dict(),
        "intentional_fail": fault.to_dict(),
    }


def main() -> int:
    """Print JSON and succeed only when both expected outcomes occur."""
    payload = run()
    print(json.dumps(payload, indent=2, sort_keys=True))
    clean = payload["clean"]
    fault = payload["intentional_fail"]
    assert isinstance(clean, dict) and isinstance(fault, dict)
    return 0 if clean["outcome"] == "PASS" and fault["outcome"] == "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
