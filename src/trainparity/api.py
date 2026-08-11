"""Frozen v0.1 public API for TrainParity's three declared checks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from trainparity.accumulation import AccumulationExecutionPlan, AccumulationRunner
from trainparity.comparison import Difference, ExactComparison, ToleranceComparison
from trainparity.outcomes import Outcome
from trainparity.process_resume import ProcessResumeRunner
from trainparity.protocols import (
    AccumulationCase,
    LossAccounting,
    ProcessExecutionPlan,
    ProcessResumeCase,
    TrainingState,
)
from trainparity.results import (
    AccumulationResult,
    ProcessResumeResult,
)
from trainparity.sample_coverage import (
    AtLeastOnce,
    ExactlyOnce,
    ExpectedPadding,
    NoCrossRankOverlap,
    SampleAnomaly,
    SampleCoverageResult,
    SampleObservation,
    SampleViolation,
    audit_rank_iterables,
    audit_sample_coverage,
)
from trainparity.version import MACHINE_REPORT_SCHEMA_VERSION

ComparisonPolicy = ExactComparison | ToleranceComparison


def check_resume(
    case: str,
    *,
    cwd: Path | None = None,
    work_dir: Path | None = None,
    report_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timeout: float = 300.0,
    temporary_root: Path | None = None,
) -> ProcessResumeResult:
    """Check one importable command-oriented resume case in fresh processes."""
    return ProcessResumeRunner(timeout=timeout, temporary_root=temporary_root).run(
        case,
        cwd=cwd,
        work_dir=work_dir,
        report_path=report_path,
        environment=environment,
    )


def check_accumulation(
    case: str,
    *,
    candidate: AccumulationExecutionPlan,
    comparison: ComparisonPolicy | None = None,
    device: str = "cpu",
    seed: int = 23,
    cwd: Path | None = None,
    report_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timeout: float = 300.0,
    temporary_root: Path | None = None,
) -> AccumulationResult:
    """Check one user-declared full-batch/accumulation equivalence relation."""
    policy = comparison if comparison is not None else ExactComparison()
    return AccumulationRunner(
        comparison=policy,
        timeout=timeout,
        temporary_root=temporary_root,
    ).run(
        case,
        candidate=candidate,
        device=device,
        seed=seed,
        cwd=cwd,
        report_path=report_path,
        environment=environment,
    )


__all__ = [
    "MACHINE_REPORT_SCHEMA_VERSION",
    "AccumulationCase",
    "AccumulationExecutionPlan",
    "AccumulationResult",
    "AtLeastOnce",
    "ComparisonPolicy",
    "Difference",
    "ExactComparison",
    "ExactlyOnce",
    "ExpectedPadding",
    "LossAccounting",
    "NoCrossRankOverlap",
    "Outcome",
    "ProcessExecutionPlan",
    "ProcessResumeCase",
    "ProcessResumeResult",
    "SampleAnomaly",
    "SampleCoverageResult",
    "SampleObservation",
    "SampleViolation",
    "ToleranceComparison",
    "TrainingState",
    "audit_rank_iterables",
    "audit_sample_coverage",
    "check_accumulation",
    "check_resume",
]
