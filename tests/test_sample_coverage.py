from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from torch.utils.data import DistributedSampler, IterableDataset, Sampler

from trainparity import (
    AtLeastOnce,
    ExactlyOnce,
    ExpectedPadding,
    NoCrossRankOverlap,
    Outcome,
    audit_sample_coverage,
)
from trainparity.sample_coverage import (
    SampleCoverageAuditor,
    SampleObservation,
    audit_rank_iterables,
)


def _distributed(length: int, world_size: int, *, drop_last: bool = False, epoch: int = 0) -> dict[int, list[list[int]]]:
    dataset = list(range(length))
    ranks: dict[int, list[list[int]]] = {}
    for rank in range(world_size):
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=17,
            drop_last=drop_last,
        )
        sampler.set_epoch(epoch)
        ranks[rank] = [list(sampler)]
    return ranks


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_exactly_once_world_sizes(world_size: int) -> None:
    result = audit_rank_iterables(
        _distributed(12, world_size),
        sample_id_extractor=lambda batch: batch,
        policy=ExactlyOnce(range(12)),
    )
    assert result.outcome is Outcome.PASS
    assert result.total_observations == result.unique_observed_ids == 12


def test_non_divisible_expected_padding_reports_repeats_and_ranks(tmp_path: Path) -> None:
    evidence = tmp_path / "padding.json"
    result = audit_rank_iterables(
        _distributed(10, 3),
        sample_id_extractor=lambda batch: batch,
        policy=ExpectedPadding(range(10), padding_count=2),
        evidence_path=evidence,
    )
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert result.outcome is Outcome.PASS
    assert result.actual_padding_count == result.expected_padding_count == 2
    assert result.repeated_id_count == 2
    assert payload["repeated_ids"]
    assert all(len(item["ranks"]) >= 1 for item in payload["repeated_ids"])
    assert payload["actual_padding_count"] == 2


def test_drop_last_reports_missing_id() -> None:
    result = audit_rank_iterables(
        _distributed(10, 3, drop_last=True),
        sample_id_extractor=lambda batch: batch,
        policy=ExactlyOnce(range(10)),
    )
    assert result.outcome is Outcome.FAIL
    assert result.missing_id_count == 1
    assert result.first_violation is not None
    assert result.first_violation.kind == "missing_id"


def test_same_rank_duplicate_is_distinct_from_cross_rank_overlap() -> None:
    observations = [
        SampleObservation("a", 0, 0, 0),
        SampleObservation("a", 0, 0, 1),
        SampleObservation("b", 1, 0, 0),
    ]
    exact = audit_sample_coverage(observations, ExactlyOnce(("a", "b")))
    no_cross = audit_sample_coverage(observations, NoCrossRankOverlap())
    assert exact.outcome is Outcome.FAIL
    assert exact.same_rank_duplicate_id_count == 1
    assert exact.cross_rank_overlap_id_count == 0
    assert exact.first_violation is not None
    assert exact.first_violation.kind == "same_rank_duplicate"
    assert no_cross.outcome is Outcome.PASS
    assert no_cross.same_rank_duplicate_id_count == 1


def test_cross_rank_overlap_has_deterministic_first_violation() -> None:
    observations = [
        SampleObservation(1, 0, 2, 0, worker=3),
        SampleObservation(2, 1, 2, 0, worker=4),
        SampleObservation(1, 1, 2, 1, worker=4),
    ]
    first = audit_sample_coverage(observations, NoCrossRankOverlap())
    second = audit_sample_coverage(observations, NoCrossRankOverlap())
    assert first.to_dict() == second.to_dict()
    assert first.outcome is Outcome.FAIL
    assert first.cross_rank_overlap_id_count == 1
    assert first.first_violation is not None
    assert first.first_violation.kind == "cross_rank_overlap"
    assert first.first_violation.rank == 1
    assert first.first_violation.worker == 4


def test_at_least_once_allows_duplicates_but_reports_them() -> None:
    observations = [
        SampleObservation(0, 0, 0, 0),
        SampleObservation(0, 1, 0, 0),
        SampleObservation(1, 0, 0, 1),
    ]
    result = audit_sample_coverage(observations, AtLeastOnce(range(2)))
    assert result.outcome is Outcome.PASS
    assert result.repeated_id_count == 1
    assert result.cross_rank_overlap_id_count == 1


def test_unexpected_id_is_not_silently_accepted() -> None:
    result = audit_sample_coverage(
        [SampleObservation(0, 0, 0, 0), SampleObservation(9, 0, 0, 1)],
        AtLeastOnce((0,)),
    )
    assert result.outcome is Outcome.FAIL
    assert result.unexpected_id_count == 1
    assert result.first_violation is not None
    assert result.first_violation.kind == "unexpected_id"


def test_expected_padding_count_must_match_declared_policy() -> None:
    result = audit_sample_coverage(
        [SampleObservation(0, 0, 0, 0), SampleObservation(0, 1, 0, 0), SampleObservation(1, 1, 0, 1)],
        ExpectedPadding((0, 1), padding_count=2),
    )
    assert result.outcome is Outcome.FAIL
    assert result.actual_padding_count == 1
    assert result.first_violation is not None
    assert result.first_violation.path == "coverage.actual_padding_count"


def test_unknown_universe_abstains_without_consuming_stream() -> None:
    def unsafe_stream() -> Iterator[SampleObservation]:
        raise AssertionError("must not consume an unknown or unbounded stream")
        yield SampleObservation(0, 0, 0, 0)

    for policy in (ExactlyOnce(None), AtLeastOnce(None), ExpectedPadding(None, 0)):
        result = audit_sample_coverage(unsafe_stream(), policy)
        assert result.outcome is Outcome.ABSTAIN
        assert result.total_observations == 0


def test_empty_universe_is_a_finite_contract_not_unknown() -> None:
    result = audit_sample_coverage([], ExactlyOnce(()))
    assert result.outcome is Outcome.PASS
    assert result.expected_id_count == 0


class _FiniteStream(IterableDataset[int]):
    def __iter__(self) -> Iterator[int]:
        yield from range(5)


def test_finite_iterable_dataset_uses_observable_contract() -> None:
    result = audit_rank_iterables(
        {0: [_FiniteStream()]},
        sample_id_extractor=lambda stream: iter(stream),
        policy=ExactlyOnce(range(5)),
    )
    assert result.outcome is Outcome.PASS


class _ReverseSampler(Sampler[int]):
    def __iter__(self) -> Iterator[int]:
        yield from (3, 2, 1, 0)

    def __len__(self) -> int:
        return 4


def test_custom_sampler_only_needs_id_extractor() -> None:
    result = SampleCoverageAuditor[list[int]](lambda batch: batch).audit(
        {0: [list(_ReverseSampler())]}, ExactlyOnce(range(4)), epoch=7
    )
    assert result.outcome is Outcome.PASS


def test_multi_epoch_shuffle_is_a_per_epoch_declared_window(tmp_path: Path) -> None:
    orders = []
    for epoch in (0, 1):
        ranks = _distributed(12, 3, epoch=epoch)
        orders.append(tuple(value for rank in sorted(ranks) for value in ranks[rank][0]))
        result = audit_rank_iterables(
            ranks,
            sample_id_extractor=lambda batch: batch,
            policy=ExactlyOnce(range(12)),
            epoch=epoch,
            evidence_path=tmp_path / f"epoch_{epoch}.json",
        )
        assert result.outcome is Outcome.PASS
        payload = json.loads((tmp_path / f"epoch_{epoch}.json").read_text(encoding="utf-8"))
        assert {occurrence["epoch"] for trace in payload["traces"] for occurrence in trace["occurrences"]} == {epoch}
    assert orders[0] != orders[1]


def test_resume_cursor_duplicate_is_visible_without_checkpoint_logic() -> None:
    before_resume = [SampleObservation(value, 0, 0, value) for value in (0, 1, 2)]
    after_resume = [SampleObservation(value, 0, 0, position) for position, value in enumerate((2, 3), start=3)]
    result = audit_sample_coverage(before_resume + after_resume, ExactlyOnce(range(4)))
    assert result.outcome is Outcome.FAIL
    assert result.first_violation is not None
    assert result.first_violation.sample_id == 2


def test_terminal_summary_is_bounded_but_machine_evidence_is_complete(tmp_path: Path) -> None:
    evidence = tmp_path / "complete.json"
    result = audit_sample_coverage([], ExactlyOnce(range(20)), max_examples=3, evidence_path=evidence)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert result.missing_id_count == 20
    assert len(result.anomaly_examples) == 3
    assert len(payload["missing_ids"]) == 20
    assert result.evidence_path == str(evidence)


@pytest.mark.parametrize(
    ("observations", "policy"),
    [
        ([SampleObservation(True, 0, 0, 0)], ExactlyOnce((1,))),
        ([SampleObservation(0, -1, 0, 0)], ExactlyOnce((0,))),
        ([], ExactlyOnce((0, 0))),
        ([], ExpectedPadding((), -1)),
    ],
)
def test_invalid_contracts_return_error(
    observations: list[SampleObservation], policy: ExactlyOnce | ExpectedPadding
) -> None:
    assert audit_sample_coverage(observations, policy).outcome is Outcome.ERROR


def test_extractor_failure_and_string_return_are_errors() -> None:
    def broken(_batch: object) -> Iterator[int]:
        raise RuntimeError("private failure details")

    failed = audit_rank_iterables({0: [object()]}, sample_id_extractor=broken, policy=ExactlyOnce((0,)))
    string = audit_rank_iterables({0: [object()]}, sample_id_extractor=lambda _batch: "secret", policy=ExactlyOnce((0,)))
    assert failed.outcome is Outcome.ERROR
    assert "private failure details" not in failed.message
    assert string.outcome is Outcome.ERROR


def test_zero_examples_and_mixed_stable_id_types() -> None:
    result = audit_sample_coverage(
        [SampleObservation("0", 0, 0, 0), SampleObservation(0, 0, 0, 1)],
        ExactlyOnce((0, "0", "missing")),
        max_examples=0,
    )
    assert result.outcome is Outcome.FAIL
    assert result.anomaly_examples == ()
