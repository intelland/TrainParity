"""Typed contracts implemented by user-owned training cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass
class TrainingState:
    """Mutable training objects and logical position owned by a case."""

    model: nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler | None = None
    step: int = 0
    scaler: object | None = None


@dataclass(frozen=True)
class ProcessExecutionPlan:
    """One framework-neutral external training-process invocation."""

    phase: str
    cwd: Path
    run_dir: Path
    end_step: int
    resume_from: Path | None = None


@runtime_checkable
class ProcessResumeCase(Protocol):
    """Project semantics required by the generic command-oriented runner."""

    name: str
    split_step: int
    total_step: int

    def command(self, plan: ProcessExecutionPlan) -> Sequence[str]:
        """Return the original project command for one execution plan."""

    def checkpoint_path(self, run_dir: Path) -> Path:
        """Return the original project checkpoint path inside ``run_dir``."""

    def observe_checkpoint(self, path: Path) -> Mapping[str, object]:
        """Select explicit comparable state from an original checkpoint."""


@dataclass(frozen=True)
class LossAccounting:
    """One differentiable loss with optional explicit reduction accounting."""

    value: torch.Tensor
    numerator: torch.Tensor | None = None
    denominator: int | float | None = None


@runtime_checkable
class AccumulationCase(Protocol):
    """User-owned semantics for one declared accumulation equivalence check."""

    equivalence: str

    def build(self, seed: int, device: str) -> TrainingState:
        """Construct verified-repeatable state in a fresh process."""

    def batch(self, device: str) -> object:
        """Return the complete ordered batch for one optimizer update."""

    def loss(self, state: TrainingState, batch: object) -> LossAccounting:
        """Compute loss and, when known, its numerator and denominator."""
