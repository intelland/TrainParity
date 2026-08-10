"""Tiny correct and faulty resume adapters used by Gate 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn

from trainparity.protocols import TrainingState
from trainparity.prototypes import ResumeCallbacks


class _Stateful(Protocol):
    def state_dict(self) -> dict[str, Any]: ...


class CorrectResumeCase:
    """A complete checkpoint adapter; this is the selected simple example."""

    def build(self, seed: int) -> TrainingState:
        torch.manual_seed(seed)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        return TrainingState(model, optimizer, scheduler)

    def train_step(self, state: TrainingState) -> None:
        state.optimizer.zero_grad()
        loss = (state.model(torch.ones(1, 1)) - torch.zeros(1, 1)).square().mean()
        loss.backward()
        state.optimizer.step()
        assert state.scheduler is not None
        state.scheduler.step()
        state.step += 1

    def save(self, state: TrainingState, path: Path) -> None:
        assert state.scheduler is not None
        scheduler = cast(_Stateful, state.scheduler)
        torch.save({"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": state.step}, path)

    def load(self, path: Path, seed: int) -> TrainingState:
        checkpoint: dict[str, Any] = torch.load(path, weights_only=True)
        state = self.build(seed)
        state.model.load_state_dict(checkpoint["model"])
        state.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and state.scheduler is not None:
            state.scheduler.load_state_dict(checkpoint["scheduler"])
        state.step = checkpoint["step"]
        return state


class MissingSchedulerStateCase(CorrectResumeCase):
    """Deliberately faulty adapter that omits scheduler checkpoint state."""

    def save(self, state: TrainingState, path: Path) -> None:
        torch.save({"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict(), "step": state.step}, path)


def make_resume_callbacks() -> ResumeCallbacks:
    """Build the evaluated factory-plus-callback alternative."""
    case = CorrectResumeCase()
    return ResumeCallbacks(
        build=case.build,
        train_step=case.train_step,
        save=case.save,
        load=case.load,
    )
