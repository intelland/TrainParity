# API reference

This page describes the supported Python API for TrainParity `0.1.0rc3`.
Signatures are verified by the release-surface tests. Names not listed here are
internal and may change without compatibility notice.

The complete name-to-defining-module table is in the
[public API inventory](public-api-inventory.md).

## Recommended top-level imports

`trainparity.__all__` contains exactly:

```text
check_resume, check_accumulation, audit_sample_coverage,
ExactlyOnce, AtLeastOnce, NoCrossRankOverlap, ExpectedPadding,
ExactComparison, ToleranceComparison, Outcome, __version__
```

Advanced supported types are imported from `trainparity.api`; they are not
re-exported from `trainparity`. `PACKAGE_VERSION`, `ExternalProcessEvidence`,
and `SampleCoverageAuditor` are implementation details, not public API.

## Outcomes and report metadata

Every check returns one of four distinct `Outcome` values:

- `PASS`: all required evidence was available and the declared relation held.
- `FAIL`: a deterministic first observed difference or policy violation was found.
- `ABSTAIN`: the requested judgment was unsupported or ambiguous.
- `ERROR`: execution, loading, serialization, or observation failed.

`FAIL` identifies a first observed divergence; it does not assert root cause.
Every `to_dict()` machine report contains integer `schema_version` and string
`trainparity_version`. Schema consumers should reject unknown versions.

## Resume equivalence

```python
def check_resume(
    case: str,
    *,
    cwd: Path | None = None,
    work_dir: Path | None = None,
    report_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timeout: float = 300.0,
    temporary_root: Path | None = None,
) -> ProcessResumeResult
```

`case` is an import string such as `package.module:Case`. `cwd` selects the
project command directory. `work_dir` preserves run files at a caller-selected
location; otherwise an isolated temporary directory is cleaned automatically.
`report_path` receives deterministic JSON. `environment` explicitly adds or
overrides child values, but reports retain only propagated key names. `timeout`
applies to child commands. `temporary_root` selects the parent for managed
temporary directories.

A resume adapter satisfies this structural protocol:

```python
class ProcessResumeCase(Protocol):
    name: str
    split_step: int
    total_step: int

    def command(self, plan: ProcessExecutionPlan) -> Sequence[str]: ...
    def checkpoint_path(self, run_dir: Path) -> Path: ...
    def observe_checkpoint(self, path: Path) -> Mapping[str, object]: ...
```

`ProcessExecutionPlan` has `phase`, `cwd`, `run_dir`, `end_step`, and optional
`resume_from` fields. `phase` is one of these four stable values:

| Phase | Meaning |
| --- | --- |
| `baseline_a` | First uninterrupted execution through `total_step`. |
| `baseline_b` | Independent uninterrupted repeat used to establish baseline self-consistency. |
| `candidate_split` | Interrupted execution through `split_step`; its checkpoint is saved and the process exits. |
| `candidate_resume` | Fresh execution that loads the staged split checkpoint and continues through `total_step`. |

`checkpoint_path(run_dir)` must return one deterministic checkpoint location
for every phase directory. After a child finishes, TrainParity uses it to find
the output checkpoint. For `candidate_resume`, TrainParity also calls it
*before* launching the resumed child, copies the `candidate_split` checkpoint
to that location, and supplies the location as `plan.resume_from`. It is
therefore a location contract, not a callback that searches only for an
already-existing file.

The adapter owns project semantics and original checkpoint commands;
TrainParity owns baseline self-consistency, continuous and interrupted plans,
process exit, fresh-process load, snapshots, comparison, timeouts, and
reporting. In-process callback boundaries catch `Exception`, not
`BaseException`: `KeyboardInterrupt` and `SystemExit` from `command()` or
`checkpoint_path()` propagate. `observe_checkpoint()` runs in a snapshot
child; its ordinary exceptions are returned as `ERROR`, while abnormal child
termination is reported as a worker `ERROR`. When `work_dir` is provided, each
phase retains `stdout.log` and `stderr.log` beneath its run directory, and
child-process error messages identify the relevant stderr path. Managed
temporary runs are cleaned and are not promised to remain available.

See the [external resume integration guide](external-resume-integration.md)
for a complete command-oriented PyTorch example, checkpoint staging diagrams,
timestamped-checkpoint wrappers, and observation recommendations.

`ProcessResumeResult` exposes `outcome`, `message`, `case`,
`first_divergent_step`, `primary_difference`, `all_differences`, process
evidence, fresh-process confirmation, propagated environment key names,
timings, snapshot IPC bytes, and maximum checkpoint bytes.

Runnable installed example:

```python
from trainparity import Outcome, check_resume

result = check_resume("trainparity.quickstarts.resume:CleanCase")
assert result.outcome is Outcome.PASS
assert result.fresh_resume_processes_distinct
```

Run the complete clean/fault pair with:

```bash
python -m trainparity.quickstarts.resume
```

## Gradient-accumulation equivalence

```python
def check_accumulation(
    case: str,
    *,
    candidate: AccumulationExecutionPlan,
    comparison: ExactComparison | ToleranceComparison | None = None,
    device: str = "cpu",
    seed: int = 23,
    cwd: Path | None = None,
    report_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timeout: float = 300.0,
    temporary_root: Path | None = None,
) -> AccumulationResult
```

The baseline performs one full-batch optimizer update. `candidate` explicitly
declares the microbatch execution; TrainParity does not assume every full batch
and accumulation plan is equivalent. `comparison=None` selects exact
comparison. A tolerance is used only when the caller constructs
`ToleranceComparison(rtol=..., atol=..., equal_nan=...)`.

`AccumulationExecutionPlan` requires `microbatch_count` and supports explicit
flags for loss scaling/accounting, optimizer or scheduler steps per
microbatch, zeroing and clipping timing, AMP timing, and final-window omission.
These flags describe a test execution; they are not inferred from observed
values.

An accumulation adapter satisfies:

```python
class AccumulationCase(Protocol):
    equivalence: str

    def build(self, seed: int, device: str) -> TrainingState: ...
    def batch(self, device: str) -> object: ...
    def loss(self, state: TrainingState, batch: object) -> LossAccounting: ...
```

`TrainingState` contains model, optimizer, optional scheduler, logical step,
and optional scaler. `LossAccounting` contains differentiable `value` plus
optional numerator and denominator. Complex batches must implement an explicit
split method on the case; the default tensor-tree splitter uses batch dimension
zero without reordering.

`AccumulationResult` exposes `outcome`, `message`, declared `equivalence`,
`first_observed_phase`, difference details, fresh process IDs,
verified-initial-state status, loss-accounting status, comparison policy, and
temporary/persisted artifact sizes.

Runnable installed example:

```python
from trainparity import Outcome, check_accumulation
from trainparity.api import AccumulationExecutionPlan

result = check_accumulation(
    "trainparity.quickstarts.accumulation:Case",
    candidate=AccumulationExecutionPlan(microbatch_count=2),
)
assert result.outcome is Outcome.PASS
assert result.verified_equal_initial_state
```

```bash
python -m trainparity.quickstarts.accumulation
```

## Sample coverage

```python
def audit_sample_coverage(
    observations: Iterable[object],
    policy: object,
    *,
    max_examples: int = 10,
    evidence_path: Path | None = None,
) -> SampleCoverageResult
```

`observations` must contain `SampleObservation(sample_id, rank, epoch,
position, worker=None)` values. `max_examples` bounds terminal/report examples.
When requested, `evidence_path` receives the complete machine-readable traces.

Supported policy constructors are:

```python
ExactlyOnce(expected_ids: Iterable[int | str] | None)
AtLeastOnce(expected_ids: Iterable[int | str] | None)
NoCrossRankOverlap()
ExpectedPadding(expected_ids: Iterable[int | str] | None, padding_count: int)
```

Exactly-once, at-least-once, and expected-padding require a reliable finite
expected universe; otherwise the result is `ABSTAIN`. A stable ID must be
semantically unique inside that universe. TrainParity validates ID
trajectories, not sample contents. Unavailable worker provenance is `None` /
JSON `null`, never worker zero. One result covers only its declared finite
observation window.

`SampleCoverageResult` exposes bounded counts, the deterministic
`first_violation`, anomaly examples, and optional evidence path. Same-rank
duplication and cross-rank overlap remain distinct.

Runnable example:

```python
from trainparity import ExactlyOnce, Outcome, audit_sample_coverage
from trainparity.api import SampleObservation

observations = [SampleObservation(i, rank=0, epoch=0, position=i) for i in range(4)]
result = audit_sample_coverage(observations, ExactlyOnce(range(4)))
assert result.outcome is Outcome.PASS
```

The complete PyTorch `DataLoader` integration in
[`examples/test_readme_case.py`](../examples/test_readme_case.py) is executed
by CI. The installed clean/fault pair is:

```bash
python -m trainparity.quickstarts.sample_coverage
```

## Advanced supported imports

`trainparity.api` supports these adapter and result objects in addition to the
recommended top-level names:

- `MACHINE_REPORT_SCHEMA_VERSION`;
- `AccumulationCase`, `AccumulationExecutionPlan`, `AccumulationResult`;
- `ComparisonPolicy`, `Difference`;
- `LossAccounting`, `TrainingState`;
- `ProcessExecutionPlan`, `ProcessResumeCase`, `ProcessResumeResult`;
- `SampleObservation`, `SampleAnomaly`, `SampleViolation`,
  `SampleCoverageResult`, and `audit_rank_iterables`.

Internal runners, snapshot backends, subprocess workers, serializers,
historical resume protocols, experiments, and Gate tooling are not supported
public API. A case must be importable in a fresh interpreter; closures and
serialized live Python objects are not accepted substitutes for the process
boundary.
