"""Explicit accumulation semantics for the pinned ImageNet recipe."""

import torch
from imagenet import main
from torch.nn import functional as F

from trainparity import LossAccounting, TrainingState


class Case:
    equivalence = "same ordered images; no stochastic augmentation; global cross-entropy mean"

    def build(self, seed: int, device: str) -> TrainingState:
        torch.manual_seed(seed)
        model = main.models.shufflenet_v2_x0_5(num_classes=4).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
        return TrainingState(model, optimizer, scheduler)

    def batch(self, device: str) -> object:
        generator = torch.Generator(device=device).manual_seed(91)
        return {
            "images": torch.randn(4, 3, 32, 32, generator=generator, device=device),
            "labels": torch.tensor([0, 1, 2, 3], device=device),
        }

    def loss(self, state: TrainingState, batch: object) -> LossAccounting:
        assert isinstance(batch, dict)
        logits = state.model(batch["images"])
        numerator = F.cross_entropy(logits, batch["labels"], reduction="sum")
        count = int(batch["labels"].numel())
        return LossAccounting(numerator / count, numerator, count)
