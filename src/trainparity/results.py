"""Public four-state result model for resume equivalence checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from trainparity.comparison import Difference
from trainparity.outcomes import Outcome


@dataclass(frozen=True)
class ProcessEvidence:
    """Process and freshly constructed object identities recorded by a worker."""

    pid: int
    model_id: int
    optimizer_id: int
    scheduler_id: int | None
    scaler_id: int | None


@dataclass(frozen=True)
class ResumeResult:
    """Outcome and evidence from one complete resume-equivalence run."""

    outcome: Outcome
    message: str
    last_matching_step: int | None = None
    first_divergent_step: int | None = None
    phase: str | None = None
    primary_difference: Difference | None = None
    all_differences: tuple[Difference, ...] = ()
    baseline_a: ProcessEvidence | None = None
    baseline_b: ProcessEvidence | None = None
    pre_save: ProcessEvidence | None = None
    post_load: ProcessEvidence | None = None
    checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        return payload


@dataclass(frozen=True)
class ExternalProcessEvidence:
    """Non-secret evidence from one original external training process."""

    phase: str
    pid: int
    elapsed_seconds: float
    returncode: int


@dataclass(frozen=True)
class ProcessResumeResult:
    """Four-state result for one command-oriented fresh-process resume test."""

    outcome: Outcome
    message: str
    case: str
    first_divergent_step: int | None = None
    primary_difference: Difference | None = None
    all_differences: tuple[Difference, ...] = ()
    processes: tuple[ExternalProcessEvidence, ...] = ()
    fresh_resume_processes_distinct: bool = False
    propagated_environment_keys: tuple[str, ...] = ()
    timing_seconds: dict[str, float] | None = None
    snapshot_ipc_bytes: int = 0
    checkpoint_max_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation without environment values."""
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        return payload

