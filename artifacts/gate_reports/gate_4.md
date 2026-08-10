# Gate 4 report

## Outcome

**PASS — recommendation: GO**

Three pinned external projects satisfy the Gate 4 product-friction criteria. Human review is required before any later gate.

## Acceptance criteria

- [x] Gate 4 implementation and evidence files: present
- [x] three distinct real external structures: projects=['ignite_mnist_engine', 'nanogpt', 'pytorch_examples_imagenet']
- [x] all clean resume cases pass: clean=3/3
- [x] one realistic fault detected per project: detected=3/3
- [x] exact commits and licenses recorded: three commit, SPDX license, and license hashes recorded
- [x] external upstream training code remains unmodified: modified_loc=0
- [x] original upstream checkpoint save/load exercised: original save and load paths recorded for all three
- [x] adapter and integration LOC recorded: adapter_locs=[24, 24, 24], median=24
- [x] integrations over 50 LOC explained: explained=['pytorch_examples_imagenet', 'nanogpt', 'ignite_mnist_engine']
- [x] minimal hand-written comparison recorded: effort and generic final-state diagnostics recorded
- [x] runtime, memory, artifact, and overhead measured: positive measurements for all three projects
- [x] full unit, contract, and integration suite: tests=76 passed, coverage=94.86%
- [x] single real M3 GPU execution recorded: gpu=NVIDIA L40S, job=58960426
- [x] accepted Gate 0-3 evidence preserved: hashes unchanged
- [x] hosted CI executes one real external case and Gate 4 verifier: run=31414864786, conclusion=success, pinned nanoGPT and Gate 4 verifier passed

## External integrations

| Project | Commit | License | Adapter LOC | Glue LOC | Upstream modified LOC | Total integration LOC | Clean | Fault | First observed divergence |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| pytorch_examples_imagenet | acc295dc7b90714f1bf47f06004fc19a7fe235c4 | BSD-3-Clause | 24 | 48 | 0 | 72 | PASS | FAIL | `scheduler.last_epoch` |
| nanogpt | 3adf61e154c3fe3fca428ad6bc3818b27a3b8291 | MIT | 24 | 57 | 0 | 81 | PASS | FAIL | `best_val_loss` |
| ignite_mnist_engine | e08ff9257ed18d8d805304e32ba85a44553195fc | BSD-3-Clause | 24 | 44 | 0 | 68 | PASS | FAIL | `lr_scheduler.last_epoch` |

Median adapter logical LOC: 24.
Shared integration logical LOC: 322.
Minimal hand-written comparator logical LOC: 12.
Tests / coverage: 76 passed / 94.86%.
Hosted CI: run `31414864786` at commit `59876afdd744ca40e2add625113320fc75168385`.

## Resource measurements

| Project | Upstream runtime (s) | Peak RSS (KiB) | Max checkpoint (bytes) | Max snapshot (bytes) | Comparison overhead |
|---|---:|---:|---:|---:|---:|
| pytorch_examples_imagenet | 24.172 | 1162372 | 11136602 | 14712711 | 0.027961% |
| nanogpt | 29.074 | 1027376 | 24051 | 24518 | 0.004816% |
| ignite_mnist_engine | 36.336 | 1110768 | 194474 | 294570 | 0.005444% |

## Hand-written comparison

The minimal control compares only final model state. Its output is either
`final model states are equal` or `final model states differ`; it does not
identify a step or state path. TrainParity reports the first observed divergence
and preserves all differences at that step. These are observations, not
root-cause claims.

## Exact commands

- `make lint`
- `make typecheck`
- `make test`
- `make build`
- `sbatch scripts/slurm_gate4_matrix.sbatch --gate 4`
- `python scripts/verify_gate.py 4`
- `git diff --check`

## Remaining limitations

- The cases use tiny generated data and one NVIDIA L40S; they measure integration friction, not training quality or scale.
- The experiment uses a correctness-first full-value snapshot backend and does not optimize snapshot size or speed.
- Gate 4 command drivers and state normalizers are experiment-only, not framework-specific production adapters.
- Ignite RunningAverage is excluded because the upstream Engine resets that reporting-only derived metric after loading; trainer, model, optimizer, and scheduler remain compared.
- PyTorch 2.6+ requires TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for the pinned Ignite example's original trusted checkpoint load call.
- Only completed-training-step resume is evaluated; distributed and accumulation behavior remain outside Gate 4.
- Every reported path is a first observed divergence, not a root-cause claim.
