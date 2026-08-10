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
