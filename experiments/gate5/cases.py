"""Small framework-neutral Gate 5 fixtures; faults live in execution plans."""

from __future__ import annotations

import torch
from torch import nn

from trainparity import LossAccounting, TrainingState


class LinearCase:
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


class MLPCase(LinearCase):
    equivalence = "same ordered tensor-tree batch; deterministic MLP with no batch-coupled layer"

    def build(self, seed: int, device: str) -> TrainingState:
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 1)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return TrainingState(model, optimizer, scheduler)


class TokenCase:
    equivalence = "same ordered masked tokens; token loss uses global numerator/denominator"

    def build(self, seed: int, device: str) -> TrainingState:
        torch.manual_seed(seed)
        model = nn.Embedding(8, 1, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return TrainingState(model, optimizer, scheduler)

    def batch(self, device: str) -> object:
        return {
            "tokens": torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0], [1, 3, 5]], device=device),
            "targets": torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.4, 0.5], [0.6, 0.0, 0.0], [0.1, 0.3, 0.5]], device=device),
            "mask": torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1]], device=device),
        }

    def loss(self, state: TrainingState, batch: object) -> LossAccounting:
        assert isinstance(batch, dict)
        prediction = state.model(batch["tokens"]).squeeze(-1)
        mask = batch["mask"].to(prediction.dtype)
        numerator = ((prediction - batch["targets"]) ** 2 * mask).sum()
        denominator = int(mask.sum().item())
        return LossAccounting(numerator / denominator, numerator, denominator)


class AmpCase(LinearCase):
    equivalence = "same ordered CUDA samples and explicit reduction under GradScaler"

    def build(self, seed: int, device: str) -> TrainingState:
        if not device.startswith("cuda"):
            raise RuntimeError("AmpCase requires CUDA")
        state = super().build(seed, device)
        state.scaler = torch.amp.GradScaler("cuda", init_scale=16.0)
        return state
