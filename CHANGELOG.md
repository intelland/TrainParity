# Changelog

## 0.1.0rc1 - 2026-08-11

- Freeze the public 0.1 API for resume equivalence, accumulation equivalence,
  and finite sample-coverage policies.
- Add schema and package versions to machine reports and three installed CPU
  examples with clean and intentional-failure outcomes.
- Add evidence-first user, API, validation, design, limitation, provenance,
  contribution, security, and release documentation.
- Add clean-environment wheel tests and source/distribution audits while
  excluding Gate experiments and accepted evidence from runtime artifacts.
- Preserve exact comparison semantics, explicit tolerance, four-state results,
  and first-observed-divergence wording.
- Replace the accidental 30-name top-level namespace with 11 recommended
  imports while retaining adapter/result types in `trainparity.api`.
- Verify installed CPU wheels on Python 3.11 with PyTorch 2.7.0, 2.10.0, and
  2.13.0; declare only the evidence-backed `torch>=2.7,<2.14` range.
- Split read-only PR CI, scheduled/manual full validation, and protected
  Trusted Publishing into separate least-privilege workflows. Publication
  remains held.

## 0.1.0.dev7 - 2026-08-11

- Add four explicit sample-coverage policies with stable-ID trajectory,
  rank/worker/epoch/position provenance, finite-universe controls, bounded
  output, and optional complete anomaly evidence.

## 0.1.0.dev6 - 2026-08-11

- Add user-declared full-batch/microbatch accumulation equivalence at a single
  optimizer-update boundary with bounded loss, gradient, optimizer/parameter,
  and scheduler observations.

## 0.1.0.dev5 - 2026-08-11

- Move framework-neutral continuous/interrupted planning, checkpoint staging,
  fresh-process execution, snapshot IPC, timeout/temp handling, deterministic
  reports, environment propagation, and four-state outcomes into production.
- Reduce the three fresh-clone user surfaces to 30, 31, and 32 logical LOC with
  zero upstream modifications and no framework-specific production adapters.
- Replace byte-at-a-time tensor-storage extraction while preserving exact raw
  bytes and `FullValueBackend` comparison semantics.
- Add Gate 4B profiling, full end-to-end/resource evidence, contract tests, a
  dedicated verifier, and hosted fresh-clone CI coverage.

## 0.1.0.dev4 - 2026-08-11

- Evaluate three exact-commit external PyTorch projects through their original
  checkpoint save/load implementations with zero upstream source changes.
- Add experiment-only adapters for ImageNet, nanoGPT, and an Ignite Engine
  recipe, with one realistic resume fault and a minimal hand-written control.
- Record adapter and glue LOC, licenses, runtime, peak memory, artifact size,
  full-value snapshot size, overhead, and every first-step difference.
- Add a real pinned nanoGPT CI integration and the Gate 4 verifier/report.

## 0.1.0.dev3 - 2026-08-11

- Add real-process continuous versus checkpoint/resume trajectory execution.
- Add initial and independent baseline-self-consistency prechecks.
- Preserve four-state runner outcomes and every difference at the first
  divergent completed-step boundary.
- Add stable batch identity, process/object evidence, formal CPU faults, and
  same-device CUDA RNG and GradScaler verification.

## 0.1.0.dev2 - 2026-08-10

- Add the Gate 2 immutable full-value snapshot reference backend.
- Canonicalize optimizer state through stable model parameter names.
- Add separate exact and explicit-tolerance comparison policies with stable
  first-observed-difference reports.
- Preserve four-state capture outcomes for unsupported and ambiguous state.

## 0.1.0.dev1 - 2026-08-10

- Add the Gate 1 installable package skeleton.
- Select an importable class/protocol resume adapter after comparing it with a
  factory-plus-callback prototype.
- Add one correct and one deliberately faulty tiny resume case.
- Add lint, type-check, test, build, CI, and Gate 1 verification workflows.
