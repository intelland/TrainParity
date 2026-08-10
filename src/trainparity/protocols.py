"""Typed contracts implemented by user-owned training cases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass
class TrainingState:
    """Mutable objects and logical position owned by a resume case."""

    model: nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler | None
    step: int = 0
    scaler: object | None = None


@dataclass(frozen=True)
class StepObservation:
    """Stable identity and user state observed after one completed step."""

    sample_ids: tuple[str | int, ...] | None = None
    batch_fingerprint: str | None = None
    extras: Mapping[str, object] = field(default_factory=dict)

    def batch_state(self) -> Mapping[str, object] | None:
        """Return one deterministic batch identity, or ``None`` if unavailable."""
        if self.sample_ids is not None:
            return {"sample_ids": list(self.sample_ids)}
        if self.batch_fingerprint is not None:
            return {"fingerprint": self.batch_fingerprint}
        return None


@runtime_checkable
class ResumeCase(Protocol):
    """Small process-importable contract for a resume-equivalence case."""

    def build(self, seed: int) -> TrainingState:
        """Construct a fresh training state deterministically from ``seed``."""

    def train_step(self, state: TrainingState) -> None:
        """Execute exactly one logical training step and update ``state.step``."""

    def save(self, state: TrainingState, path: Path) -> None:
        """Save every object needed to continue at the next logical step."""

    def load(self, path: Path, seed: int) -> TrainingState:
        """Construct and restore a state from ``path`` for the supplied seed."""


@runtime_checkable
class ResumeExecutionCase(ResumeCase, Protocol):
    """Gate 3 extension that exposes stable post-step observations."""

    def observe(self, state: TrainingState) -> StepObservation:
        """Describe the most recently completed batch and supported extra state."""
