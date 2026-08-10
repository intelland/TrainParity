"""Pytest-friendly assertions for resume equivalence results."""

from trainparity.outcomes import Outcome
from trainparity.results import ResumeResult


def assert_resume_equivalent(result: ResumeResult) -> None:
    """Assert PASS while preserving full result evidence on failure."""
    if result.outcome is not Outcome.PASS:
        raise AssertionError(result.to_dict())
