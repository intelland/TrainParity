"""Small framework-neutral accumulation fixture for fresh-process tests."""

from __future__ import annotations

import torch
from torch import nn

from trainparity.protocols import LossAccounting, TrainingState


class LinearCase:
    """A deterministic full-batch versus microbatch equivalence case."""

    equivalence = "same four ordered samples; full mean equals summed numerator/denominator"

    def build(self, seed: int, device: str) -> TrainingState:
        torch.manual_seed(seed)
        model = nn.Linear(1, 1, bias=False, device=device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.125, momentum=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return TrainingState(model, optimizer, scheduler)

    def batch(self, device: str) -> object:
        return {
            "x": torch.tensor([[1.0], [2.0], [4.0], [8.0]], device=device),
            "y": torch.tensor([[0.5], [1.0], [2.0], [4.0]], device=device),
        }

    def loss(self, state: TrainingState, batch: object) -> LossAccounting:
        assert isinstance(batch, dict)
        residual = state.model(batch["x"]) - batch["y"]
        numerator = residual.square().sum()
        return LossAccounting(numerator / residual.numel(), numerator, residual.numel())
