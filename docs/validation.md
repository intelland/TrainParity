# Validation evidence

This page records validation for the TrainParity 0.1.0 release. It is not a
universal detection rate. The fixtures are intentionally small and the
external repositories are pinned. A passing row demonstrates only the declared
relation on the recorded environment.

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

| Project structure | Upstream commit | License | Clean | Fault | User LOC |
| --- | --- | --- | --- | --- | ---: |
| PyTorch examples ImageNet classifier | `acc295dc7b90714f1bf47f06004fc19a7fe235c4` | BSD-3-Clause | PASS | scheduler state detected | 30 |
| nanoGPT sequence model | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | MIT | PASS | training control state detected | 31 |
| Ignite MNIST Engine recipe | `e08ff9257ed18d8d805304e32ba85a44553195fc` | BSD-3-Clause | PASS | scheduler state detected | 32 |

These tiny L40S fixtures are product-surface checks, not claims about all uses
of those projects.

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

## Release and environment record

The accepted `0.1.0rc5` source passed Ruff, strict Mypy, 182 tests at 90.73%
measured source coverage, the complete README integration, all three
quickstarts, wheel/sdist build, Twine validation, and release-surface checks in
GitHub Actions run `31798915064`. Release workflow run `31816180948` built
the distributions once, smoke-tested the exact wheel outside the source tree,
rechecked artifact hashes, and published those files through Trusted
Publishing. A fresh Python 3.11 environment then installed the public wheel and
verified package version `0.1.0rc5`, machine-report schema 2, and all three
clean/fault quickstarts.

### Stable 0.1.0 main CI

GitHub Actions run `32237392040` completed successfully on the accepted stable
source at `main@2c8fb257fa8b133578502361d29161851da1ff3a`.

### Stable 0.1.0 release workflow

Release workflow run `32238085358` completed successfully. It built the wheel
and sdist once, validated them with Twine, smoke-tested the exact wheel outside
the source tree, verified installed version `0.1.0` and machine-report schema
2, ran all three quickstarts, rechecked artifact hashes, and published the same
`dist/` contents through protected-environment Trusted Publishing. This is
release evidence, not a reproducible-build guarantee.

### Official 0.1.0 artifacts

| Artifact | SHA-256 |
| --- | --- |
| `trainparity-0.1.0-py3-none-any.whl` | `2f3afa565da36406cdd39c70d8e2ea0bb8cad593f7685336dbed67bde628b952` |
| `trainparity-0.1.0.tar.gz` | `6fb88e390c299c0aa98cbd25c704217a5694f2be8766275d9fe660ac6dc8d994` |

### Public PyPI verification

An outside-repository Python 3.11.15 environment installed PyTorch
2.13.0+cpu and `trainparity==0.1.0` from public package indexes. The import
resolved from that environment's `site-packages`, with package version `0.1.0`
and machine-report schema 2. Installed quickstarts produced:

- Resume: clean `PASS`; intentional `FAIL` at `scheduler.last_epoch`.
- Accumulation: clean `PASS`; intentional `FAIL` at
  `loss_accounting.denominator`.
- Sample coverage: clean `PASS`; intentional `FAIL` at
  `coverage.same_rank_duplicate`.

The independently downloaded public PyPI wheel and sdist hashes matched the
corresponding release-workflow hashes above.

## Compatibility boundary

Tested compatibility for this stable release is deliberately narrow:

| Component | Tested |
| --- | --- |
| Python | CPython 3.11.15 |
| CPU PyTorch | installed wheels with 2.7.0+cpu, 2.10.0+cpu, and 2.13.0+cpu |
| GPU PyTorch | 2.7.0+cu126 |
| CPU | Linux x86-64; each version ran all three clean/fault quickstarts outside the repository |
| GPU | same-device A100 80GB PCIe and L40S fixtures, CUDA 12.6 |
| Process model | single training process plus fresh child processes |

Each CPU row built and installed a wheel with normal dependency resolution in
a fresh Python 3.11 environment, then ran resume, accumulation, and
sample-coverage clean/fault quickstarts outside the repository. All three rows
passed without source-tree fallback. The declared runtime dependency is
`torch>=2.7,<2.14`; the upper bound does not assert compatibility with an
untested PyTorch 2.14.

Untested Python, PyTorch, CUDA, operating-system, accelerator, distributed, and
large-model combinations are not covered by this matrix.

Historical release validation is recorded evidence, while current CI
continuously verifies the supported product and packaging surface. The
scheduled and manually dispatched Validation workflow verifies the declared CPU
compatibility matrix.

## Coverage measurement boundary

The source-coverage configuration omits modules whose execution is not
represented correctly by the parent pytest process:

- `accumulation_worker.py`, `process_worker.py`, and `worker.py` run in real
  fresh subprocesses. `tests/test_accumulation.py`,
  `tests/test_process_resume.py`, and `tests/test_runner.py` assert their exit,
  IPC, error, identity, and result behavior. The current coverage run does not
  combine subprocess coverage files, so counting these files as uncovered
  would misstate the tested process boundary.
- `protocols.py` contains structural protocols plus data carriers exercised by
  the runners; protocol placeholder bodies are not executable behavior.
- `prototypes.py` and `trainparity/examples/` are outside the documented 0.1
  public API, and the examples package is excluded from the wheel.

Public facade, importing, comparison, orchestration, serialization, reporting,
and quickstart modules are included in measured source coverage. Subprocess
coverage is documented separately rather than silently presented as
parent-process line coverage.
