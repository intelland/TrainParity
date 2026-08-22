# Changelog

## 0.1.0 - 2026-08-19

- Promote the behavior, public API, and machine-report schema verified through
  `0.1.0rc5` to the first non-prerelease release.
- Retain machine-report schema 2, the 11-name recommended top-level API, and
  the documented advanced API without adding comparison semantics.
- Retain exact-by-default comparison, explicit user-selected tolerance,
  four-state outcomes, and first-observed-divergence reporting.
- Preserve the validated Python/PyTorch range and framework-neutral scope.

## 0.1.0rc5 - 2026-08-14

- Preserve actionable accumulation setup-error causes and publish requested
  machine reports through the normal deterministic finalization path.
- Publish accumulation reports as byte-stable UTF-8 artifacts whose embedded
  size agrees with the returned result and actual file across platforms.
- Distinguish sample-coverage rank-iterable failures from extractor callback
  and output-consumption failures, including rank, attempted batch index, and
  underlying exception details.
- Preserve accumulation comparison semantics, sample-coverage policies,
  public API signatures, and machine-report schema 2.

## 0.1.0rc4 - 2026-08-13

- Add an explicit `comparison=` policy to command-oriented resume checks while
  preserving exact comparison as the default.
- Record the resume comparison policy and declared tolerance values in machine
  reports, whose global schema is now version 2.
- Report numerical error magnitudes for exact floating-point and complex
  tensor mismatches without weakening exact semantics.

## 0.1.0rc3 - 2026-08-13

- Convert ordinary command-oriented resume adapter failures from `command()`,
  `checkpoint_path()`, and `observe_checkpoint()` into phase-specific
  `Outcome.ERROR` results while preserving `KeyboardInterrupt` and
  `SystemExit` propagation.
- Document the four stable external resume phases, pre-launch checkpoint
  staging, deterministic and timestamped checkpoint locations, preserved child
  logs, recommended observations, and expected orchestration cost.
- Add a complete external PyTorch resume integration guide and verify that its
  copyable example runs and the guide enters the source distribution.
- Preserve public API signatures, comparison semantics, four-state meanings,
  compatibility claims, baseline self-consistency, and fresh-process behavior.

## 0.1.0rc2 - 2026-08-12

- Allow scheduler-free accumulation cases to construct `TrainingState` without
  explicitly passing `scheduler=None`.
- Document an explicit CPU-only PyTorch installation path before installing
  TrainParity.
- Document the development dependencies required by the source-checkout pytest
  example.
- Preserve the existing checks, public API, four-state outcomes, comparison
  semantics, declared PyTorch dependency range, and validated support claims.

## 0.1.0rc1 - 2026-08-11

- Freeze the public 0.1 API for resume equivalence, accumulation equivalence,
  and finite sample-coverage policies.
- Add schema and package versions to machine reports and three installed CPU
  examples with clean and intentional-failure outcomes.
- Add user, API, validation, design, limitation, contribution, security, and
  release documentation.
- Add clean-environment wheel tests and source/distribution audits while
  validating the packaged release surface in clean environments.
- Preserve exact comparison semantics, explicit tolerance, four-state results,
  and first-observed-divergence wording.
- Replace the accidental 30-name top-level namespace with 11 recommended
  imports while retaining adapter/result types in `trainparity.api`.
- Verify installed CPU wheels on Python 3.11 with PyTorch 2.7.0, 2.10.0, and
  2.13.0; declare only the evidence-backed `torch>=2.7,<2.14` range.
- Split read-only CI and protected Trusted Publishing into separate
  least-privilege workflows.
