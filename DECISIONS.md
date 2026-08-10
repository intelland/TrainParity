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

## D-0006: Recommend Gate 0 GO, conditional on human review

- **Status:** Proposed
- **Date:** 2026-08-10
- **Decision:** Recommend that a human approve Gate 0 because the explicit A/B
  prototype precisely located three core faults for which TrainCheck produced
  no fault-specific evidence after clean-control correction.
- **Reason:** The difference follows from the execution model and output
  contract, not UI or naming. Gate 1 remains unauthorized until approval.
