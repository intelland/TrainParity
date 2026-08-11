# Validation evidence

This is TrainParity's reproducible validation suite, not a universal detection
rate. The fixtures are intentionally small and the external repositories are
pinned. A passing row demonstrates only the declared relation on the recorded
environment.

## Resume equivalence

The reference matrix ran three clean CPU repetitions and three clean
same-device GPU repetitions with zero false positives. It detected 13 of 13
stable faults and matched the expected first observed component in all 13:
model, optimizer, scheduler, Python/NumPy/CPU/CUDA RNG, data cursor, logical
step, optimizer group, hidden module state, and GradScaler cases. Every
checkpoint save and load crossed distinct process IDs.

The GPU fixture used Python 3.11.15, PyTorch 2.7.0+cu126, CUDA 12.6, and one
NVIDIA A100 80GB PCIe. Cross-GPU numerical comparison was not performed.

Three external integrations exercised original checkpoint implementations at
exact commits with zero upstream modified LOC:

| Project structure | Upstream commit | License | Clean | Fault | User LOC after Gate 4B |
| --- | --- | --- | --- | --- | ---: |
| PyTorch examples ImageNet classifier | `acc295dc7b90714f1bf47f06004fc19a7fe235c4` | BSD-3-Clause | PASS | scheduler state detected | 30 |
| nanoGPT sequence model | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | MIT | PASS | training control state detected | 31 |
| Ignite MNIST Engine recipe | pinned in Gate 4 evidence | BSD-3-Clause | PASS | scheduler state detected | 32 |

These tiny L40S fixtures are product-surface checks, not claims about all uses
of those projects. Detailed pins, license hashes, checkpoint paths, LOC, and
timing are preserved in `artifacts/gate_reports/gate_4.json` and
`gate_4b.json`.

## Accumulation equivalence

Fresh-process baseline and candidate executions began from verified-equal
state. Three CPU clean cases and one same-device NVIDIA L40S clean case passed.
All eight required fault fixtures failed at their expected bounded phase:
missing scaling, variable-length mean-of-means, optimizer step per microbatch,
scheduler step per microbatch, misplaced zeroing, misplaced gradient clipping,
AMP unscale/GradScaler timing, and an incomplete final window.

The two pinned product surfaces required 38 logical LOC for ImageNet and 37 for
nanoGPT with no upstream edits or framework-specific production branch. The
nanoGPT fixture observes the tied parameter in full model and gradient state
but excludes its ambiguous alias from optimizer state; strict production
canonicalization otherwise returns `ABSTAIN`.

## Sample coverage

Seventeen of seventeen CPU matrix rows matched their expected outcome across
world sizes 1, 2, 3, and 4, non-divisible lengths, `drop_last`, declared
padding, missing IDs, same-rank duplicates, cross-rank overlap, a custom
sampler, finite `IterableDataset`, unknown-universe `ABSTAIN`, and multi-epoch
shuffle observations. `SequentialSampler` and `DistributedSampler` product
surfaces required 15 and 18 logical LOC with zero upstream changes.

The expected universe and ID extractor are user declarations. The auditor does
not inspect sample contents or make claims beyond one finite observation
window.

## Test and environment boundary

Gate 6 concluded with 134 passing tests and 92.46% measured source coverage.
Gate 7 reruns the complete suite, every accepted Gate verifier, archive audits,
and all three examples from a built wheel outside the repository. The Gate 7
report records the final counts and hosted CI run.

Tested compatibility for this release candidate is deliberately narrow:

| Component | Tested |
| --- | --- |
| Python | CPython 3.11.15 |
| PyTorch | 2.7.0+cu126 on M3; hosted CPU version recorded by Gate 7 CI |
| CPU | Linux x86-64 |
| GPU | same-device A100 80GB PCIe and L40S fixtures, CUDA 12.6 |
| Process model | single training process plus fresh child processes |

Untested Python, PyTorch, CUDA, operating-system, accelerator, distributed, and
large-model combinations are not covered by this matrix.

An optional same-device replay on the configured M3 cluster is:
`sbatch scripts/slurm_gpu_matrix.sbatch --gate 3`. It requires the documented
cluster environment and is not a portable launcher.
