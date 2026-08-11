"""Run the deterministic CPU-only Gate 6 sample-coverage matrix."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DistributedSampler, IterableDataset, Sampler

from trainparity import (
    AtLeastOnce,
    ExactlyOnce,
    ExpectedPadding,
    NoCrossRankOverlap,
    audit_sample_coverage,
)
from trainparity.sample_coverage import (
    SampleCoverageResult,
    SampleObservation,
    audit_rank_iterables,
)


class _CustomSampler(Sampler[int]):
    def __iter__(self) -> Iterator[int]:
        yield from (4, 2, 0, 3, 1)

    def __len__(self) -> int:
        return 5


class _FiniteIterable(IterableDataset[int]):
    def __iter__(self) -> Iterator[int]:
        yield from range(5)


def _distributed(length: int, world_size: int, *, drop_last: bool = False, epoch: int = 0) -> dict[int, list[list[int]]]:
    dataset = list(range(length))
    ranks = {}
    for rank in range(world_size):
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=True, seed=23, drop_last=drop_last)
        sampler.set_epoch(epoch)
        ranks[rank] = [list(sampler)]
    return ranks


def _row(
    name: str,
    expected: str,
    result: SampleCoverageResult,
    requirements: Iterable[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "expected": expected,
        "requirements": sorted(requirements),
        "result": result.to_dict(),
    }


def run(output: Path) -> dict[str, Any]:
    """Run all required policy, sampler, and anomaly fixtures on CPU."""
    raw = output.parent / "machine_evidence"
    rows = []
    for world_size in (1, 2, 3, 4):
        result = audit_rank_iterables(
            _distributed(12, world_size),
            sample_id_extractor=lambda batch: batch,
            policy=ExactlyOnce(range(12)),
            evidence_path=raw / f"world_size_{world_size}.json",
        )
        rows.append(_row(f"world_size_{world_size}", "PASS", result, (f"world_size_{world_size}",)))
    non_divisible = audit_rank_iterables(
        _distributed(10, 3),
        sample_id_extractor=lambda batch: batch,
        policy=ExpectedPadding(range(10), 2),
        evidence_path=raw / "non_divisible_padding.json",
    )
    rows.append(_row("non_divisible_padding", "PASS", non_divisible, ("non_divisible", "padding_duplicate", "expected_padding")))
    drop_last = audit_rank_iterables(
        _distributed(10, 3, drop_last=True),
        sample_id_extractor=lambda batch: batch,
        policy=ExactlyOnce(range(10)),
        evidence_path=raw / "drop_last.json",
    )
    rows.append(_row("drop_last_missing", "FAIL", drop_last, ("drop_last", "missing_ids")))
    same_rank = [SampleObservation(0, 0, 0, 0), SampleObservation(0, 0, 0, 1), SampleObservation(1, 1, 0, 0)]
    rows.append(_row("same_rank_duplicate", "FAIL", audit_sample_coverage(same_rank, ExactlyOnce(range(2)), evidence_path=raw / "same_rank.json"), ("same_rank_duplication",)))
    rows.append(_row("same_rank_allowed_by_no_cross", "PASS", audit_sample_coverage(same_rank, NoCrossRankOverlap()), ("same_rank_distinct_condition",)))
    cross_rank = [SampleObservation(0, 0, 0, 0), SampleObservation(1, 0, 0, 1), SampleObservation(0, 1, 0, 0)]
    rows.append(_row("cross_rank_overlap", "FAIL", audit_sample_coverage(cross_rank, NoCrossRankOverlap(), evidence_path=raw / "cross_rank.json"), ("cross_rank_overlap",)))
    missing = [SampleObservation(value, 0, 0, value) for value in range(4)]
    rows.append(_row("missing_id", "FAIL", audit_sample_coverage(missing, AtLeastOnce(range(5))), ("missing_ids", "at_least_once")))
    rows.append(_row("padding_count_mismatch", "FAIL", audit_sample_coverage(cross_rank, ExpectedPadding(range(2), 2)), ("padding_duplicate", "expected_padding")))
    rows.append(_row("custom_sampler", "PASS", audit_rank_iterables({0: [list(_CustomSampler())]}, sample_id_extractor=lambda batch: batch, policy=ExactlyOnce(range(5))), ("custom_sampler",)))
    rows.append(_row("finite_iterable_dataset", "PASS", audit_rank_iterables({0: [_FiniteIterable()]}, sample_id_extractor=lambda stream: iter(stream), policy=ExactlyOnce(range(5))), ("finite_iterable_dataset",)))
    rows.append(_row("unknown_universe", "ABSTAIN", audit_sample_coverage([], ExactlyOnce(None)), ("unknown_universe_abstain",)))
    rows.append(_row("at_least_once_repeats", "PASS", audit_sample_coverage(cross_rank, AtLeastOnce(range(2))), ("at_least_once",)))
    for epoch in (0, 1):
        result = audit_rank_iterables(
            _distributed(12, 3, epoch=epoch),
            sample_id_extractor=lambda batch: batch,
            policy=ExactlyOnce(range(12)),
            epoch=epoch,
            evidence_path=raw / f"shuffle_epoch_{epoch}.json",
        )
        rows.append(_row(f"shuffle_epoch_{epoch}", "PASS", result, ("multi_epoch_shuffle",)))
    payload = {
        "schema_version": 1,
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    payload = run(arguments.output)
    mismatches = [row for row in payload["rows"] if row["result"]["outcome"] != row["expected"]]
    print(json.dumps({"rows": len(payload["rows"]), "mismatches": len(mismatches)}, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
