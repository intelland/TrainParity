"""Gate 1-only alternate API prototype used for evidence collection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from trainparity.protocols import TrainingState


@dataclass(frozen=True)
class ResumeCallbacks:
    """Factory-plus-callback alternative evaluated but not selected."""

    build: Callable[[int], TrainingState]
    train_step: Callable[[TrainingState], None]
    save: Callable[[TrainingState, Path], None]
    load: Callable[[Path, int], TrainingState]


CallbackFactory = Callable[[], ResumeCallbacks]

