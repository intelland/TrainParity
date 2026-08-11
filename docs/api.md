# Public API for 0.1

The supported import surface is `trainparity.api`. The same names are re-exported
from `trainparity` and listed in its `__all__`. The 0.1 release line freezes the
objects below; anything else is internal and can change without compatibility
notice.

## Entry points

- `check_resume` and `check_accumulation`
- `audit_sample_coverage` and `audit_rank_iterables`
- `PACKAGE_VERSION` and `MACHINE_REPORT_SCHEMA_VERSION`

## Declared cases and plans

- `ProcessResumeCase`, `ProcessExecutionPlan`, `ExternalProcessEvidence`
- `AccumulationCase`, `AccumulationExecutionPlan`, `LossAccounting`,
  `TrainingState`
- `SampleObservation` and `SampleCoverageAuditor`

## Policies, results, and values

- `ExactComparison`, `ToleranceComparison`, `ComparisonPolicy`, `Difference`
- `ExactlyOnce`, `AtLeastOnce`, `NoCrossRankOverlap`, `ExpectedPadding`
- `ProcessResumeResult`, `AccumulationResult`, `SampleCoverageResult`
- `SampleAnomaly`, `SampleViolation`, and `Outcome`

Machine reports include integer `schema_version` and string
`trainparity_version` fields. Schema version 1 is the 0.1 report contract.
Consumers must reject unsupported schema versions rather than guessing their
meaning.

Internal runners (`ProcessResumeRunner`, `AccumulationRunner`), snapshot
backends (`FullValueBackend`), subprocess workers, serializers, experiment
packages, and Gate verification scripts are deliberately outside this public
surface. Some historical top-level imports remain temporarily available so
accepted evidence can be replayed, but omission from `__all__` means they are
not stable API.

TrainParity uses ordinary Python protocols, not framework adapters. A case must
be importable in a fresh interpreter, because a closure or serialized live
object would weaken the real process-boundary test.
