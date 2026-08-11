# Gate 4B — Production Integration Surface

## Outcome

**GO** — stop after Gate 4B for human review. Gate 5 was not started.

Generic baseline/candidate planning, checkpoint staging, fresh-process execution,
snapshot IPC, deterministic reporting, timeout/temp-directory behavior, explicit
environment propagation, and PASS/FAIL/ABSTAIN/ERROR handling now live in the
framework-neutral TrainParity production package. No project-specific production
adapter or framework branch was added.

## Fresh-clone product surface

| Project | Adapter LOC | Glue LOC | Total user LOC | Upstream modified LOC | Clean | Fault | Total/normal wall | Persisted artifacts (bytes) |
|---|---:|---:|---:|---:|---|---|---:|---:|
| pytorch_examples_imagenet | 17 | 13 | 30 | 0 | PASS | FAIL | 5.261x | 2937 |
| nanogpt | 18 | 13 | 31 | 0 | PASS | FAIL | 5.202x | 2911 |
| ignite_mnist_engine | 19 | 13 | 32 | 0 | PASS | FAIL | 4.949x | 2914 |

Median total user LOC is 31; the
functionally closer hand-written fresh-process baseline is 116 logical LOC and is
project-specific. TrainParity retains exact model/optimizer/scheduler/RNG state,
fresh-process evidence, first-observed-divergence diagnostics, and four-state
result semantics with less user code in every project.

## ImageNet snapshot profile

The pre-optimization profile found byte-at-a-time storage iteration, not tensor
cloning or repeated materialization, as the dominant cost. FullValueBackend remains
the correctness reference; no fingerprint or collision-bearing backend was added.
The full snapshot path fell from 22.278513s to
0.183875s (121.16x), while byte-for-byte
compatibility tests preserve comparison semantics.

## Injected faults

| Project | Class | First observed divergence | Downstream parameter divergence |
|---|---|---|---|
| pytorch_examples_imagenet | control-state | scheduler.last_epoch | false |
| nanogpt | trajectory-affecting | best_val_loss | true |
| ignite_mnist_engine | control-state | lr_scheduler.last_epoch | false |

These are first observed divergences, not root-cause claims.

## Verification

- Full lint, type-check, tests, coverage, build, and Gate 0-4 verifier replay: PASS.
- Three exact-commit external fresh clones on NVIDIA L40S: PASS.
- Explicit child environment propagation, including `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`, is contract-tested; reports contain keys only, never environment values.
- Hosted GitHub Actions: success (run 31437964382).
- Accepted Gate 0-4 evidence hashes and the recorded user-document hash are unchanged.

## Scope

No production framework adapter, distributed support, dashboard, service, backend
semantic weakening, Gate 5 work, or root-cause claim was introduced.
