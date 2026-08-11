"""Explicit sample-coverage policies over user-observed stable sample IDs."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeAlias, TypeVar

from trainparity.outcomes import Outcome
from trainparity.version import add_report_metadata

SampleId: TypeAlias = int | str
BatchT = TypeVar("BatchT")


@dataclass(frozen=True)
class ExactlyOnce:
    """Require every declared expected ID exactly once in one audit window."""

    expected_ids: Iterable[SampleId] | None
    name: ClassVar[str] = "exactly_once"


@dataclass(frozen=True)
class AtLeastOnce:
    """Require every declared expected ID at least once in one audit window."""

    expected_ids: Iterable[SampleId] | None
    name: ClassVar[str] = "at_least_once"


@dataclass(frozen=True)
class NoCrossRankOverlap:
    """Require each observed ID to occur on no more than one rank."""

    name: ClassVar[str] = "no_cross_rank_overlap"


@dataclass(frozen=True)
class ExpectedPadding:
    """Require a finite ID universe and an exact number of repeated observations."""

    expected_ids: Iterable[SampleId] | None
    padding_count: int
    name: ClassVar[str] = "expected_padding"


CoveragePolicy: TypeAlias = ExactlyOnce | AtLeastOnce | NoCrossRankOverlap | ExpectedPadding


@dataclass(frozen=True)
class SampleObservation:
    """One stable sample ID with optional distributed provenance."""

    sample_id: SampleId
    rank: int
    epoch: int
    position: int
    worker: int | None = None


@dataclass(frozen=True)
class SampleViolation:
    """The deterministic first policy violation observed by an audit."""

    kind: str
    path: str
    sample_id: SampleId | None
    rank: int | None
    worker: int | None
    epoch: int | None
    position: int | None


@dataclass(frozen=True)
class SampleAnomaly:
    """A bounded anomaly example for terminal-safe reporting."""

    kind: str
    sample_id: SampleId
    occurrence_count: int
    ranks: tuple[int, ...]
    workers: tuple[int, ...]
    epochs: tuple[int, ...]


@dataclass(frozen=True)
class SampleCoverageResult:
    """Four-state, bounded summary of one declared sample-coverage policy."""

    outcome: Outcome
    policy: str
    message: str
    total_observations: int = 0
    unique_observed_ids: int = 0
    expected_id_count: int | None = None
    missing_id_count: int = 0
    unexpected_id_count: int = 0
    repeated_id_count: int = 0
    same_rank_duplicate_id_count: int = 0
    cross_rank_overlap_id_count: int = 0
    actual_padding_count: int | None = None
    expected_padding_count: int | None = None
    first_violation: SampleViolation | None = None
    anomaly_examples: tuple[SampleAnomaly, ...] = ()
    evidence_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible bounded summary."""
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        return add_report_metadata(payload)


@dataclass(frozen=True)
class _Analysis:
    observations: tuple[SampleObservation, ...]
    traces: dict[SampleId, tuple[SampleObservation, ...]]
    expected_ids: tuple[SampleId, ...] | None
    missing_ids: tuple[SampleId, ...]
    unexpected_ids: tuple[SampleId, ...]
    repeated_ids: tuple[SampleId, ...]
    same_rank_duplicates: tuple[tuple[SampleId, int], ...]
    cross_rank_overlaps: tuple[SampleId, ...]
    actual_padding_count: int


def _id_key(sample_id: SampleId) -> tuple[str, str]:
    return (type(sample_id).__name__, repr(sample_id))


def _valid_id(value: object) -> bool:
    return isinstance(value, (int, str)) and not isinstance(value, bool)


def _policy_name(policy: object) -> str:
    name = getattr(policy, "name", None)
    return name if isinstance(name, str) else "unsupported"


def _expected_ids(policy: CoveragePolicy) -> Iterable[SampleId] | None:
    if isinstance(policy, NoCrossRankOverlap):
        return None
    return policy.expected_ids


def _materialize_expected(policy: CoveragePolicy) -> tuple[tuple[SampleId, ...] | None, str | None]:
    values = _expected_ids(policy)
    if values is None:
        return None, None
    try:
        materialized = tuple(values)
    except Exception as error:
        return None, f"expected ID universe could not be read: {type(error).__name__}"
    if any(not _valid_id(value) for value in materialized):
        return None, "expected ID universe contains an unsupported ID; use int or str"
    if len(set(materialized)) != len(materialized):
        return None, "expected ID universe contains duplicate IDs"
    return tuple(sorted(materialized, key=_id_key)), None


def _validate_observation(observation: SampleObservation) -> str | None:
    if not _valid_id(observation.sample_id):
        return "observed sample ID is unsupported; use int or str"
    if isinstance(observation.rank, bool) or observation.rank < 0:
        return "rank must be a non-negative integer"
    if isinstance(observation.epoch, bool) or observation.epoch < 0:
        return "epoch must be a non-negative integer"
    if isinstance(observation.position, bool) or observation.position < 0:
        return "position must be a non-negative integer"
    if observation.worker is not None and (isinstance(observation.worker, bool) or observation.worker < 0):
        return "worker must be None or a non-negative integer"
    return None


def _analyse(
    observations: tuple[SampleObservation, ...], expected_ids: tuple[SampleId, ...] | None
) -> _Analysis:
    mutable_traces: dict[SampleId, list[SampleObservation]] = defaultdict(list)
    for observation in observations:
        mutable_traces[observation.sample_id].append(observation)
    traces = {
        sample_id: tuple(mutable_traces[sample_id])
        for sample_id in sorted(mutable_traces, key=_id_key)
    }
    observed = set(traces)
    expected = None if expected_ids is None else set(expected_ids)
    missing = () if expected is None else tuple(sorted(expected - observed, key=_id_key))
    unexpected = () if expected is None else tuple(sorted(observed - expected, key=_id_key))
    repeated = tuple(sample_id for sample_id, trace in traces.items() if len(trace) > 1)
    same_rank: list[tuple[SampleId, int]] = []
    cross_rank: list[SampleId] = []
    for sample_id, trace in traces.items():
        ranks: dict[int, int] = defaultdict(int)
        for observation in trace:
            ranks[observation.rank] += 1
        same_rank.extend((sample_id, rank) for rank, count in sorted(ranks.items()) if count > 1)
        if len(ranks) > 1:
            cross_rank.append(sample_id)
    padding = sum(max(len(trace) - 1, 0) for sample_id, trace in traces.items() if expected is None or sample_id in expected)
    return _Analysis(
        observations,
        traces,
        expected_ids,
        missing,
        unexpected,
        repeated,
        tuple(same_rank),
        tuple(cross_rank),
        padding,
    )


def _first_duplicate(
    observations: tuple[SampleObservation, ...], *, cross_rank_only: bool = False
) -> SampleViolation | None:
    first_rank: dict[SampleId, int] = {}
    seen_on_rank: set[tuple[SampleId, int]] = set()
    for observation in observations:
        key = (observation.sample_id, observation.rank)
        prior_rank = first_rank.setdefault(observation.sample_id, observation.rank)
        if cross_rank_only and prior_rank != observation.rank:
            return _violation("cross_rank_overlap", "coverage.cross_rank_overlap", observation)
        if not cross_rank_only and key in seen_on_rank:
            return _violation("same_rank_duplicate", "coverage.same_rank_duplicate", observation)
        if not cross_rank_only and prior_rank != observation.rank:
            return _violation("cross_rank_overlap", "coverage.cross_rank_overlap", observation)
        seen_on_rank.add(key)
    return None


def _first_unexpected(
    observations: tuple[SampleObservation, ...], expected_ids: tuple[SampleId, ...]
) -> SampleViolation | None:
    expected = set(expected_ids)
    for observation in observations:
        if observation.sample_id not in expected:
            return _violation("unexpected_id", "coverage.unexpected_ids", observation)
    return None


def _violation(kind: str, path: str, observation: SampleObservation) -> SampleViolation:
    return SampleViolation(
        kind,
        path,
        observation.sample_id,
        observation.rank,
        observation.worker,
        observation.epoch,
        observation.position,
    )


def _earlier(
    observations: tuple[SampleObservation, ...],
    left: SampleViolation | None,
    right: SampleViolation | None,
) -> SampleViolation | None:
    if left is None:
        return right
    if right is None:
        return left
    left_index = next(
        index
        for index, observation in enumerate(observations)
        if observation.sample_id == left.sample_id
        and observation.rank == left.rank
        and observation.epoch == left.epoch
        and observation.position == left.position
    )
    right_index = next(
        index
        for index, observation in enumerate(observations)
        if observation.sample_id == right.sample_id
        and observation.rank == right.rank
        and observation.epoch == right.epoch
        and observation.position == right.position
    )
    return left if left_index <= right_index else right


def _outcome(policy: CoveragePolicy, analysis: _Analysis) -> tuple[Outcome, str, SampleViolation | None]:
    expected = analysis.expected_ids
    unexpected = None if expected is None else _first_unexpected(analysis.observations, expected)
    if isinstance(policy, ExactlyOnce):
        duplicate = _first_duplicate(analysis.observations)
        first = _earlier(analysis.observations, unexpected, duplicate)
        if first is None and analysis.missing_ids:
            first = SampleViolation("missing_id", "coverage.missing_ids", analysis.missing_ids[0], None, None, None, None)
        if first is not None:
            return Outcome.FAIL, "exactly_once policy was violated", first
        return Outcome.PASS, "every expected sample ID was observed exactly once", None
    if isinstance(policy, AtLeastOnce):
        first = unexpected
        if first is None and analysis.missing_ids:
            first = SampleViolation("missing_id", "coverage.missing_ids", analysis.missing_ids[0], None, None, None, None)
        if first is not None:
            return Outcome.FAIL, "at_least_once policy was violated", first
        return Outcome.PASS, "every expected sample ID was observed at least once", None
    if isinstance(policy, NoCrossRankOverlap):
        first = _first_duplicate(analysis.observations, cross_rank_only=True)
        if first is not None:
            return Outcome.FAIL, "no_cross_rank_overlap policy was violated", first
        return Outcome.PASS, "no sample ID was observed on more than one rank", None
    failures = bool(analysis.missing_ids or analysis.unexpected_ids or analysis.actual_padding_count != policy.padding_count)
    first = unexpected
    if first is None and analysis.missing_ids:
        first = SampleViolation("missing_id", "coverage.missing_ids", analysis.missing_ids[0], None, None, None, None)
    if first is None and analysis.actual_padding_count != policy.padding_count:
        first = SampleViolation("padding_count", "coverage.actual_padding_count", None, None, None, None, None)
    if failures:
        return Outcome.FAIL, "expected_padding policy was violated", first
    return Outcome.PASS, "observed repetitions match the declared padding policy", None


def _anomalies(analysis: _Analysis, limit: int) -> tuple[SampleAnomaly, ...]:
    kinds: dict[SampleId, list[str]] = defaultdict(list)
    for sample_id in analysis.missing_ids:
        kinds[sample_id].append("missing_id")
    for sample_id in analysis.unexpected_ids:
        kinds[sample_id].append("unexpected_id")
    for sample_id in analysis.repeated_ids:
        kinds[sample_id].append("repeated_id")
    for sample_id, _rank in analysis.same_rank_duplicates:
        kinds[sample_id].append("same_rank_duplicate")
    for sample_id in analysis.cross_rank_overlaps:
        kinds[sample_id].append("cross_rank_overlap")
    rows = []
    for sample_id in sorted(kinds, key=_id_key)[:limit]:
        trace = analysis.traces.get(sample_id, ())
        rows.append(
            SampleAnomaly(
                "+".join(sorted(set(kinds[sample_id]))),
                sample_id,
                len(trace),
                tuple(sorted({item.rank for item in trace})),
                tuple(sorted({item.worker for item in trace if item.worker is not None})),
                tuple(sorted({item.epoch for item in trace})),
            )
        )
    return tuple(rows)


def _full_evidence(
    policy: CoveragePolicy, outcome: Outcome, analysis: _Analysis, first: SampleViolation | None
) -> dict[str, Any]:
    return add_report_metadata({
        "outcome": outcome.value,
        "policy": _policy_name(policy),
        "first_violation": None if first is None else asdict(first),
        "expected_ids": analysis.expected_ids,
        "missing_ids": analysis.missing_ids,
        "unexpected_ids": analysis.unexpected_ids,
        "repeated_ids": [
            {
                "sample_id": sample_id,
                "ranks": sorted({item.rank for item in analysis.traces[sample_id]}),
                "occurrence_count": len(analysis.traces[sample_id]),
            }
            for sample_id in analysis.repeated_ids
        ],
        "same_rank_duplicates": [
            {"sample_id": sample_id, "rank": rank}
            for sample_id, rank in analysis.same_rank_duplicates
        ],
        "cross_rank_overlaps": [
            {
                "sample_id": sample_id,
                "ranks": sorted({item.rank for item in analysis.traces[sample_id]}),
            }
            for sample_id in analysis.cross_rank_overlaps
        ],
        "actual_padding_count": analysis.actual_padding_count,
        "traces": [
            {
                "sample_id": sample_id,
                "occurrences": [asdict(item) for item in trace],
            }
            for sample_id, trace in analysis.traces.items()
        ],
    })


def audit_sample_coverage(
    observations: Iterable[object],
    policy: object,
    *,
    max_examples: int = 10,
    evidence_path: Path | None = None,
) -> SampleCoverageResult:
    """Audit observations without launching ranks, workers, or training processes."""
    name = _policy_name(policy)
    if isinstance(max_examples, bool) or max_examples < 0:
        return SampleCoverageResult(Outcome.ERROR, name, "max_examples must be a non-negative integer")
    if not isinstance(policy, (ExactlyOnce, AtLeastOnce, NoCrossRankOverlap, ExpectedPadding)):
        return SampleCoverageResult(Outcome.ERROR, name, "unsupported sample-coverage policy")
    if isinstance(policy, ExpectedPadding) and (isinstance(policy.padding_count, bool) or policy.padding_count < 0):
        return SampleCoverageResult(Outcome.ERROR, name, "padding_count must be a non-negative integer")
    expected, expected_error = _materialize_expected(policy)
    if expected_error is not None:
        return SampleCoverageResult(Outcome.ERROR, name, expected_error)
    if not isinstance(policy, NoCrossRankOverlap) and expected is None:
        return SampleCoverageResult(
            Outcome.ABSTAIN,
            name,
            "a reliable finite expected ID universe is required for this policy",
        )
    try:
        materialized = tuple(observations)
    except Exception as error:
        return SampleCoverageResult(Outcome.ERROR, name, f"observations could not be read: {type(error).__name__}")
    validated: list[SampleObservation] = []
    for value in materialized:
        if not isinstance(value, SampleObservation):
            return SampleCoverageResult(Outcome.ERROR, name, "observations must be SampleObservation values")
        validation_error = _validate_observation(value)
        if validation_error is not None:
            return SampleCoverageResult(Outcome.ERROR, name, validation_error)
        validated.append(value)
    analysis = _analyse(tuple(validated), expected)
    outcome, message, first = _outcome(policy, analysis)
    target = None
    if evidence_path is not None:
        try:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(
                json.dumps(_full_evidence(policy, outcome, analysis, first), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            target = str(evidence_path)
        except OSError as error:
            return SampleCoverageResult(Outcome.ERROR, name, f"machine evidence could not be written: {type(error).__name__}")
    return SampleCoverageResult(
        outcome=outcome,
        policy=name,
        message=message,
        total_observations=len(materialized),
        unique_observed_ids=len(analysis.traces),
        expected_id_count=None if expected is None else len(expected),
        missing_id_count=len(analysis.missing_ids),
        unexpected_id_count=len(analysis.unexpected_ids),
        repeated_id_count=len(analysis.repeated_ids),
        same_rank_duplicate_id_count=len({sample_id for sample_id, _rank in analysis.same_rank_duplicates}),
        cross_rank_overlap_id_count=len(analysis.cross_rank_overlaps),
        actual_padding_count=analysis.actual_padding_count if isinstance(policy, ExpectedPadding) else None,
        expected_padding_count=policy.padding_count if isinstance(policy, ExpectedPadding) else None,
        first_violation=first,
        anomaly_examples=_anomalies(analysis, max_examples),
        evidence_path=target,
    )


def audit_rank_iterables(
    rank_iterables: Mapping[int, Iterable[BatchT]],
    *,
    sample_id_extractor: Callable[[BatchT], Iterable[SampleId]],
    policy: CoveragePolicy,
    epoch: int = 0,
    max_examples: int = 10,
    evidence_path: Path | None = None,
) -> SampleCoverageResult:
    """Collect stable IDs from rank-labelled iterables and audit the declared policy."""
    observations: list[SampleObservation] = []
    try:
        for rank, iterable in sorted(rank_iterables.items()):
            if isinstance(rank, bool) or rank < 0:
                return SampleCoverageResult(Outcome.ERROR, _policy_name(policy), "rank must be a non-negative integer")
            position = 0
            for batch in iterable:
                extracted = sample_id_extractor(batch)
                if isinstance(extracted, (str, bytes)):
                    return SampleCoverageResult(Outcome.ERROR, _policy_name(policy), "sample_id_extractor must return an iterable of IDs")
                for sample_id in extracted:
                    observations.append(SampleObservation(sample_id, rank, epoch, position))
                    position += 1
    except Exception as error:
        return SampleCoverageResult(Outcome.ERROR, _policy_name(policy), f"sample ID extraction failed: {type(error).__name__}")
    return audit_sample_coverage(
        observations,
        policy,
        max_examples=max_examples,
        evidence_path=evidence_path,
    )


@dataclass(frozen=True)
class SampleCoverageAuditor(Generic[BatchT]):
    """Reusable bound ID extractor for repeated sample-coverage audits."""

    sample_id_extractor: Callable[[BatchT], Iterable[SampleId]]
    max_examples: int = 10

    def audit(
        self,
        rank_iterables: Mapping[int, Iterable[BatchT]],
        policy: CoveragePolicy,
        *,
        epoch: int = 0,
        evidence_path: Path | None = None,
    ) -> SampleCoverageResult:
        """Audit one finite observation window under an explicit policy."""
        return audit_rank_iterables(
            rank_iterables,
            sample_id_extractor=self.sample_id_extractor,
            policy=policy,
            epoch=epoch,
            max_examples=self.max_examples,
            evidence_path=evidence_path,
        )
