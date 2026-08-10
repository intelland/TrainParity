"""Four-state outcomes shared by capture and comparison operations."""

from enum import Enum


class Outcome(str, Enum):
    """A non-Boolean result that preserves unsupported and infrastructure states."""

    PASS = "PASS"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"
    ERROR = "ERROR"

