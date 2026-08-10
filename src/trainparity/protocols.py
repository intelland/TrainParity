"""Typed contracts implemented by user-owned training cases."""

from __future__ import annotations

from dataclasses import dataclass
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

