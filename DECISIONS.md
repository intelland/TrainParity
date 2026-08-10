# TrainParity Decisions

## D-0001: Execute one acceptance gate at a time

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** Gate 0 is the only active scope. Work stops after its reports
  and verification are complete, pending explicit human approval.
- **Reason:** The project has evidence-based go/no-go boundaries. Building the
  library before validating differentiation would invalidate that process.

## D-0002: Keep runtime work on M3 under the MP25 boundary

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** Local Windows is used only for repository editing and Git.
  Environments, third-party checkouts, caches, experiments, and generated logs
  live under `/scratch/mp25/jwuu0254/zxh/TrainParity` on M3.
- **Reason:** This is the repository operating contract and prevents local and
  shared-home environment drift.

## D-0003: Treat competitor evidence as an isolated experiment

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** OrderLab/TrainCheck is installed or checked out only in an
  isolated project-owned M3 environment. TrainParity may observe its behavior
  but must not copy its source code.
- **Reason:** Gate 0 requires actual evidence while preserving licensing and
  product-independence boundaries.

## D-0004: Pin a CPU-only Gate 0 evidence environment

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** Gate 0 evidence uses Python 3.11.15, PyTorch 2.13.0+cpu, and
  TrainCheck 0.1.2 in `/scratch/mp25/jwuu0254/zxh/TrainParity/envs/gate0`.
  Ruff 0.16.2 and Mypy 2.3.0 are verification-only dependencies.
- **Reason:** Gate 0 requires reproducible competitor execution but does not
  justify CUDA downloads or a production dependency decision.

## D-0005: Require clean-control correction for inferred-invariant evidence

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** A TrainCheck fault is counted as detected only when its failed
  invariant signature multiset contains evidence absent from a second clean
  target run. Raw failure counts are reported but are not sufficient.
- **Reason:** Three fixtures initially appeared detected, but every violation
  was reproduced by the clean control. Counting them would fabricate evidence.

## D-0006: Accept Gate 0 and authorize Gate 1

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** The human reviewer accepted Gate 0 and authorized Gate 1 only,
  based on the structural distinction between inferred-invariant checking and
  explicit A/B differential testing, the clean-control-corrected competitor
  results, and stable four-fault prototype evidence.
- **Reason:** The difference follows from the execution model and output
  contract, not UI or naming. Gate 0 evidence must remain unchanged, and first
  observed divergence must not be reinterpreted as root cause.

## D-0007: Run instrumented competitor copies outside the repository

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** The TrainCheck runner copies each tiny entry script into its
  case runtime directory and executes every CLI phase with that runtime as the
  working directory.
- **Reason:** TrainCheck generates instrumented Python files and tool logs beside
  inputs/current working directories. Runtime copies keep those artifacts under
  `$PROJECT_ROOT/outputs/gate0` and preserve a clean Git checkout.

## D-0008: Limit the active scope to Gate 1

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** Build only the installable skeleton, API prototypes, examples,
  engineering checks, and Gate 1 evidence. Stop before Gate 2 for human review.
- **Reason:** The human authorization explicitly permits only Gate 1. Snapshot
  normalization, comparison, and full resume orchestration belong to later
  gates and would invalidate the staged acceptance process if implemented now.

## D-0009: Select the class/protocol adapter API

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** The public Gate 1 adapter is an importable zero-argument class
  structurally checked against `ResumeCase`. The evaluated
  factory-plus-callback dataclass remains a prototype and is not public API.
- **Reason:** One `module:ClassName` identifies construction and all four
  required behaviors, works in a fresh process without `cloudpickle`, and
  avoids four callback wiring points. The selected example is 28 logical lines.

## D-0010: Use PyTorch as the sole production dependency

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** Declare only `torch>=2.5` at runtime. Build, Ruff, Mypy, pytest,
  and coverage remain development extras. Do not add NumPy merely to suppress
  PyTorch's optional-integration warning in a minimal environment.
- **Reason:** Public state types directly contain PyTorch modules, optimizers,
  and schedulers. PyTorch is therefore an honest required dependency; the
  Gate 1 implementation needs no other runtime package.

## D-0011: Pin the Gate 1 evidence environment

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** Gate 1 evidence uses Python 3.11.15 and PyTorch 2.13.0+cpu in
  `/scratch/mp25/jwuu0254/zxh/TrainParity/envs/gate1`. All subsequent build
  temporary files explicitly use the project-owned `tmp` directory.
- **Reason:** A fresh isolated CPU environment validates install metadata,
  examples, and wheel behavior without relying on Gate 0's editable state or
  paths outside the MP25 project boundary.

## D-0012: Accept Gate 1 and authorize Gate 2 only

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** The human reviewer accepted Gate 1 and authorized only the
  deterministic snapshot, canonicalization, comparison, and first-observed-
  difference work defined by Gate 2.
- **Reason:** The class/protocol adapter met the 30-line and process import
  criteria and distinguished the clean and faulty probes. Production resume
  orchestration and later equivalence checks remain unauthorized.

## D-0013: Carry forward hosted CI confirmation

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** Preserve the fact that the hosted GitHub Actions conclusion for
  commit `ae75212` could not be read through the connected GitHub App. Confirm
  it, or report a precise access blocker, no later than Gate 2 review.
- **Reason:** The equivalent complete M3 command chain was accepted as Gate 1
  evidence, but it does not independently establish GitHub-hosted execution.
