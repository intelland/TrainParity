"""Minimal Counter baseline used only for the Gate 6 product comparison."""

from collections import Counter
from collections.abc import Iterable


def counter_coverage(ids: Iterable[int], expected: Iterable[int]) -> dict[str, list[int]]:
    """Return only missing and duplicate IDs without distributed semantics."""
    counts = Counter(ids)
    universe = set(expected)
    return {
        "missing": sorted(universe - counts.keys()),
        "duplicates": sorted(sample_id for sample_id, count in counts.items() if count > 1),
    }

