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
- **Decision:** Preserve the initial App/CLI access failure and the eventual
  independent confirmation that commit `ae75212` completed GitHub Actions run
  `31394676144` with conclusion `success` before Gate 2 review.
- **Reason:** The equivalent complete M3 command chain was accepted as Gate 1
  evidence, but it did not independently establish GitHub-hosted execution. A
  read-only REST request using the local Git credential helper resolved the
  carry-forward without exposing or persisting credentials.

## D-0014: Use an immutable full-value reference backend

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** `FullValueBackend` freezes tensor values into immutable CPU
  bytes while retaining original shape, dtype, device, and gradient metadata.
  `capture_snapshot` accepts a backend protocol rather than requiring this
  storage representation for every future implementation.
- **Reason:** Immutable bytes break mutable aliases and provide a simple Gate 2
  correctness oracle without hard-coding materialized snapshots as the only
  possible backend architecture.

## D-0015: Canonicalize optimizer state only through unique parameter names

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** Optimizer groups preserve declared ordering, while all state is
  nested under names from `model.named_parameters(remove_duplicate=False)`.
  Missing names, aliases, duplicate occurrences, and unmapped state keys return
  `ABSTAIN`; memory and serialized parameter IDs are never public paths.
- **Reason:** Parameter names are the only stable and human-readable mapping
  available at this boundary. Ambiguity must not become a false `PASS`.

## D-0016: Keep exact and tolerance semantics as separate policies

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** `ExactComparison` checks structure, metadata, scalar float bits,
  and tensor bytes. `ToleranceComparison` requires explicit finite non-negative
  `rtol`/`atol` and `equal_nan`, applying tolerance only to numeric values after
  structural and metadata equality.
- **Reason:** Separate classes prevent hidden tolerance widening and make signed
  zero, NaN, Inf, empty values, and tensor metadata behavior explicit.

## D-0017: Test optional NumPy RNG capture without adding a runtime dependency

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** NumPy is a development extra for Gate 2 tests, not a production
  dependency. Capture includes NumPy RNG only when NumPy is installed.
- **Reason:** The product contract requires optional NumPy RNG support, while
  PyTorch remains the sole mandatory production dependency.

## D-0018: Pin a clean Gate 2 verification environment

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** Final Gate 2 evidence runs in
  `/scratch/mp25/jwuu0254/zxh/TrainParity/envs/gate2` with Python 3.11.15,
  PyTorch 2.13.0+cpu, NumPy 2.4.6, Ruff 0.16.2, Mypy 2.3.0, pytest 9.1.1,
  and coverage 7.15.4.
- **Reason:** A dedicated environment demonstrates the package, optional NumPy
  RNG capture, coverage, and build independently of prior Gate environments.

## D-0019: Accept Gate 2 and authorize Gate 3 only

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** The human reviewer accepted Gate 2 and authorized only the
  Resume Equivalence MVP, real fresh-process boundary, trajectory alignment,
  formal CPU fault matrix, and same-device GPU validation defined by Gate 3.
- **Reason:** Snapshot and comparison semantics met all machine criteria. Resume
  execution correctness is the next explicit go/no-go boundary; later checks
  and performance work remain unauthorized.

## D-0020: Isolate intermediate Gate work from main-branch CI email

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** Develop and synchronize Gate 3 on the `gate-3` branch. Run the
  full M3 CPU/GPU verifier there, then fast-forward `main` exactly once after a
  passing final checkpoint.
- **Reason:** The CI workflow intentionally triggers only on `main` pushes.
  Keeping intermediate checkpoints off `main` avoids failure-notification email
  while preserving Git synchronization and checkpoint history.

## D-0021: Define Gate 3 around completed-step fresh-process trajectories

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** Snapshot `step=N` is the state after exactly N completed
  optimizer updates. Compare two independent continuous baselines and initial
  states before attribution. The checkpoint writer exits before a new ordinary
  Python process constructs and restores model, optimizer, scheduler, scaler,
  RNG, data position, gradients, and declared extra state.
- **Reason:** This alignment makes the loaded split boundary observable, keeps
  nondeterminism as `ABSTAIN`, and proves a real rather than simulated process
  boundary.

## D-0022: Treat data identity as a Gate 3 prerequisite

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** Every completed step must expose sample IDs or a deterministic
  batch fingerprint. Missing identity returns `ABSTAIN`. Stable path ordering
  places batch identity first, so the cursor-offset fixture is first observed
  at `batch.sample_ids[0]` on the next consumed step.
- **Reason:** Model or optimizer differences caused by the wrong batch are
  downstream observations; the data trajectory must be checked first without
  claiming a root cause.

## D-0023: Restore serialized CUDA RNG bytes on CPU

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** Move each loaded CUDA RNG ByteTensor to CPU before calling
  `torch.cuda.set_rng_state_all`. Preserve the failed job `58957648` as `ERROR`
  evidence and accept corrected same-A100 job `58957857` as the formal result.
- **Reason:** `map_location=cuda` otherwise maps RNG tensors to CUDA, while the
  PyTorch RNG restoration API requires CPU ByteTensors. This was a load error,
  not a training parity failure.

## D-0024: Accept Gate 3 and authorize Gate 4 only

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** The human reviewer accepted Gate 3 and authorized only the
  three-project product-friction evaluation defined by Gate 4 and its additional
  constraints. Gate 5 accumulation work remains unauthorized.
- **Reason:** The fresh-process Resume Equivalence MVP met clean, fault,
  diagnostic, four-state, GPU, repeatability, coverage, CI, and evidence
  preservation requirements. Real external project effort is the next Go/No-Go.

## D-0025: Evaluate Gate 4 on a dedicated branch

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** Develop and run Gate 4 on `gate-4`; only a final verified
  checkpoint may be integrated into `main`.
- **Reason:** Real-project setup is expected to expose compatibility failures.
  Keeping intermediate work off `main` prevents avoidable CI notification email.

## D-0026: Keep all Gate 4 integrations external and experiment-only

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** Pin `pytorch/examples`, `karpathy/nanoGPT`, and
  `pytorch/ignite` as external M3 checkouts, execute their original checkpoint
  save/load paths, and keep command drivers and state normalizers outside the
  production package. Modify zero tracked upstream lines.
- **Reason:** This preserves real-project friction and prevents Gate 4 evidence
  code from becoming premature framework-specific production adapters.

## D-0027: Recommend GO after Gate 4 and stop for review

- **Status:** Proposed for human review
- **Date:** 2026-08-11
- **Decision:** Recommend `GO` based on 3/3 clean passes, 3/3 fault detections,
  median adapter LOC 24, zero upstream modifications, original checkpoint
  execution, and path-level diagnostics that outperform the minimal final-model
  control. Do not start Gate 5.
- **Reason:** The product-friction threshold is met, while the documented glue
  cost, tiny-data scope, one-GPU scope, and correctness-first backend remain
  explicit limitations for the human reviewer.

## D-0028: Reopen Gate 4 for a friction audit only

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** Rework only the Gate 4 product-friction evidence. Demonstrate
  each integration from a fresh pinned clone, count all user-visible glue,
  measure the complete workflow, and compare with a closer hand-written
  fresh-process resume test. Do not start Gate 5 or change production behavior.
- **Reason:** The original technical results were positive, but the reported
  adapter and comparator-only measurements did not establish the full cost a
  user would actually bear.

## D-0029: Recommend REWORK after the Gate 4 friction audit

- **Status:** Proposed for human review
- **Date:** 2026-08-11
- **Decision:** Keep TrainParity as a technical GO candidate, but recommend
  `REWORK` rather than Gate 5 progression. The three adapters are 13, 13, and 15
  logical lines, while complete user-required integration totals are 266, 275,
  and 271 lines and end-to-end multipliers are 15.91x, 4.03x, and 3.85x.
- **Reason:** The correctness signal and diagnostics survive fresh-clone
  reproduction, but command-oriented external use currently requires substantial
  orchestration outside the production API. Expanding that API is not authorized
  by this audit and must be a separate human product decision.

## D-0030: Move only generic process orchestration into production for Gate 4B

- **Status:** Proposed for human review
- **Date:** 2026-08-11
- **Decision:** Add one framework-neutral command-oriented process runner and
  snapshot worker. Keep original commands, checkpoint paths, and observed-state
  selection in small explicit project adapters; keep fixtures, fault injection,
  measurements, cloning, and reporting under Gate 4B experiments. Retain
  `FullValueBackend` as the exact correctness reference and stop before Gate 5.
- **Reason:** Fresh-clone user integration falls from 266-275 LOC to 30-32 LOC
  without upstream edits, framework branches, weaker comparison, or loss of
  fresh-process/four-state/first-divergence behavior.

## D-0031: Accept Gate 4B and authorize Gate 5 only

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** The human reviewer accepted Gate 4B and authorized only the
  bounded accumulation-equivalence work in Gate 5. Gate 6 sample-coverage work
  remains unauthorized.
- **Reason:** The framework-neutral production integration surface met its LOC,
  fresh-clone, runtime, exact-state, four-state, environment, and preservation
  requirements.

## D-0032: Make accumulation equivalence explicit and bounded

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** Treat full-batch versus microbatch accumulation as a
  user-declared relation over one optimizer-update boundary. Observe only loss
  accounting, completed gradients, optimizer state, parameter update, and
  scheduler/scaler state in separate fresh processes.
- **Reason:** This detects the specified accumulation faults without claiming
  universal equivalence or creating a general event-tracing platform.

## D-0033: Recommend GO after Gate 5 and stop for review

- **Status:** Proposed for human review
- **Date:** 2026-08-11
- **Decision:** Recommend `GO` based on zero clean false positives, detection of
  all eight required faults, verified-equal fresh-process starts, the same-device
  L40S result, two sub-50-LOC pinned project checks, and explicit BatchNorm and
  ambiguous-optimizer controls. Do not start Gate 6.
- **Reason:** The bounded observations distinguish loss accounting, gradient,
  optimizer, parameter, and scheduler state without asserting universal
  accumulation equivalence or expanding into general tracing.

## D-0034: Accept Gate 5 and authorize optional Gate 6 only

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** The human reviewer accepted Gate 5 and authorized only the
  bounded sample-coverage inclusion evaluation in Gate 6. Gate 6 may conclude
  `INCLUDE_MODULE`, `REWORK_MODULE`, or `OMIT_MODULE`; release work remains
  unauthorized.
- **Reason:** Resume and accumulation already form a viable product. Sample
  coverage should be included only if explicit policies and structured anomaly
  evidence add meaningful value beyond a short `Counter` script.

## D-0035: Audit observations without owning distributed execution

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** Gate 6 consumes stable sample IDs and declared rank, worker, and
  epoch provenance. It does not launch ranks, workers, DDP, Slurm, NCCL, or
  training. Exactly-once and at-least-once claims require an explicit finite
  expected universe; unknown exactly-once universes return `ABSTAIN`.
- **Reason:** This preserves observable coverage semantics and four-state
  reporting without expanding TrainParity into a distributed training system.

## D-0036: Recommend inclusion of the bounded sample-coverage module

- **Status:** Proposed for human review
- **Date:** 2026-08-11
- **Decision:** Recommend `INCLUDE_MODULE` after Gate 6 and stop before Gate 7.
  Retain only the four explicit policies, stable-ID inputs, bounded summaries,
  optional complete evidence, and four-state outcomes implemented in Gate 6.
- **Reason:** Two standard PyTorch sampler integrations require 15 and 18 user
  logical LOC with zero upstream edits. Compared with the 11-line flat
  `Counter` baseline, the module demonstrates seven structural benefits while
  all 17 formal CPU rows, 134 tests, historical verifiers, and hosted CI pass.

## D-0037: Accept Gate 6 and freeze the v0.1 feature set

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** Include the bounded sample-coverage module and authorize only
  Gate 7 release-candidate preparation. Freeze v0.1 to resume equivalence,
  gradient-accumulation equivalence, and sample-coverage policies.
- **Reason:** Sample coverage adds explicit distributed provenance and policy
  semantics beyond a flat `Counter` while retaining a small user surface. The
  next task is honest packaging and documentation, not feature expansion.

## D-0038: Define stable IDs and finite audit claims conservatively

- **Status:** Accepted
- **Date:** 2026-08-11
- **Decision:** A stable sample ID must be semantically unique within the
  declared expected universe. Unavailable worker provenance is `None`/JSON
  `null`, never rank-local worker 0. An audit proves only its declared finite
  observation window and validates ID trajectories rather than sample content.
- **Reason:** These boundaries prevent IDs, unavailable provenance, finite
  streams, and shuffle observations from being interpreted as stronger claims
  than the recorded evidence supports.

## D-0039: Freeze a narrow versioned public API for 0.1

- **Status:** Accepted for Gate 7 verification
- **Date:** 2026-08-11
- **Decision:** The stable API is the explicit facade in `trainparity.api` and
  its top-level `__all__` re-exports. Internal runners, snapshot backends,
  subprocess workers, experiments, and Gate tools are not stable public API.
  Machine reports use schema version 1 and package version `0.1.0rc1`.
- **Reason:** A small declared surface permits honest compatibility while
  retaining historical imports only for accepted-evidence replay.

## D-0040: Exclude development evidence from release distributions

- **Status:** Accepted for Gate 7 verification
- **Date:** 2026-08-11
- **Decision:** Preserve Gate reports, recorded experiments, development plans,
  and the user's remote-development document in the repository, but exclude
  them from both wheel and source distribution. Ship only runtime code and
  user-facing documentation.
- **Reason:** Auditability and a minimal runtime artifact are compatible when
  the packaging boundary is explicit and machine-verified.

## D-0041: Pin release-candidate compatibility to tested versions

- **Status:** Accepted for Gate 7 verification
- **Date:** 2026-08-11
- **Decision:** Declare Python `>=3.11,<3.12` and PyTorch `>=2.7,<2.8` for the
  0.1 release candidate. Document only the exact CPU, CUDA, and GPU environments
  actually exercised; do not imply future-version support.
- **Reason:** Open-ended dependency metadata would contradict the conservative
  tested-compatibility claim required at the release gate.

## D-0042: Recommend the audited 0.1 release candidate for human review

- **Status:** Proposed for final human review
- **Date:** 2026-08-11
- **Decision:** Gate 7 is `PASS` with recommendation
  `READY_FOR_HUMAN_REVIEW`. Stop without publishing, tagging, creating a GitHub
  release, changing repository visibility, renaming, merging the draft PR, or
  beginning a later Gate.
- **Reason:** The stable API, versioned reports, installed examples, exact
  compatibility claims, documentation, clean wheel, distribution audit,
  accepted-evidence hashes, user document hash, M3 release check, and hosted CI
  all pass with no release blocker.
