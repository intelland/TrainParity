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

