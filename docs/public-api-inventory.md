# Public API inventory

This inventory records the Gate 7I release surface and defining modules. The
recommended top-level namespace contains 11 names; `trainparity.api` contains
26 supported names including the overlapping top-level objects.

## Recommended `trainparity` names

| Name | Defining module |
|---|---|
| `check_resume` | `trainparity.api` |
| `check_accumulation` | `trainparity.api` |
| `audit_sample_coverage` | `trainparity.sample_coverage` |
| `ExactlyOnce` | `trainparity.sample_coverage` |
| `AtLeastOnce` | `trainparity.sample_coverage` |
| `NoCrossRankOverlap` | `trainparity.sample_coverage` |
| `ExpectedPadding` | `trainparity.sample_coverage` |
| `ExactComparison` | `trainparity.comparison` |
| `ToleranceComparison` | `trainparity.comparison` |
| `Outcome` | `trainparity.outcomes` |
| `__version__` | `trainparity.__init__`, sourced from `trainparity.version` |

## Additional supported `trainparity.api` names

| Name | Defining module |
|---|---|
| `MACHINE_REPORT_SCHEMA_VERSION` | `trainparity.version` |
| `AccumulationCase` | `trainparity.protocols` |
| `AccumulationExecutionPlan` | `trainparity.accumulation` |
| `AccumulationResult` | `trainparity.results` |
| `ComparisonPolicy` | `trainparity.api` |
| `Difference` | `trainparity.comparison` |
| `LossAccounting` | `trainparity.protocols` |
| `ProcessExecutionPlan` | `trainparity.protocols` |
| `ProcessResumeCase` | `trainparity.protocols` |
| `ProcessResumeResult` | `trainparity.results` |
| `SampleAnomaly` | `trainparity.sample_coverage` |
| `SampleCoverageResult` | `trainparity.sample_coverage` |
| `SampleObservation` | `trainparity.sample_coverage` |
| `SampleViolation` | `trainparity.sample_coverage` |
| `TrainingState` | `trainparity.protocols` |
| `audit_rank_iterables` | `trainparity.sample_coverage` |

The other ten names in `trainparity.api.__all__` are the same policies,
comparisons, outcome, and check functions already listed as recommended
top-level imports.

`PACKAGE_VERSION`, `ExternalProcessEvidence`, and `SampleCoverageAuditor`
remain implementation objects but are not in either public `__all__`.
