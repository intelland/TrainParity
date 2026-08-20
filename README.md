# TrainParity

[![CI](https://github.com/intelland/TrainParity/actions/workflows/ci.yml/badge.svg)](https://github.com/intelland/TrainParity/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trainparity.svg)](https://pypi.org/project/trainparity/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/intelland/TrainParity/blob/main/LICENSE)

**Differential tests for PyTorch training semantics.**

Turn assumptions about checkpoint resume and gradient accumulation — and policies about sample coverage — into executable checks.

For resume and gradient accumulation, TrainParity executes the declared comparison. For sample coverage, it audits observed stable IDs against a declared policy. On `FAIL`, it reports the first observed divergence or policy violation.

**[PyPI](https://pypi.org/project/trainparity/)** ·
**[Quickstarts](https://github.com/intelland/TrainParity/tree/main/src/trainparity/quickstarts)** ·
**[API](https://github.com/intelland/TrainParity/blob/main/docs/api.md)** ·
**[Validation](https://github.com/intelland/TrainParity/blob/main/docs/validation.md)** ·
**[Design](https://github.com/intelland/TrainParity/blob/main/docs/design.md)**

---

## What do you want to verify?

| Check                            | Question                                                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Resume parity**                | Does save → process exit → reload preserve the declared training state relative to uninterrupted execution? |
| **Gradient accumulation parity** | Does a declared microbatch plan produce the intended full-batch update at the observed phases?              |
| **Sample coverage**              | Do stable sample IDs satisfy the declared finite-window coverage policy across ranks?                       |

TrainParity returns four distinct outcomes:

| Outcome   | Meaning                                                                    |
| --------- | -------------------------------------------------------------------------- |
| `PASS`    | The declared observations satisfied the comparison or policy.              |
| `FAIL`    | An observed difference or policy violation was found.                      |
| `ABSTAIN` | The requested judgment could not be justified from the available evidence. |
| `ERROR`   | Execution or observation could not complete.                               |

A `FAIL` localizes the **first observed divergence or policy violation**. It is not a root-cause claim.

---

## Installation

```bash
pip install trainparity
```

To pin the current stable release:

```bash
pip install trainparity==0.1.0
```

Package metadata currently requires:

```text
Python >=3.11,<3.12
PyTorch >=2.7,<2.14
```

See [Compatibility](#compatibility) for the explicitly validated environments.

---

## 30-second quickstart

The installed package ships three small CPU quickstarts:

```bash
python -m trainparity.quickstarts.resume
python -m trainparity.quickstarts.accumulation
python -m trainparity.quickstarts.sample_coverage
```

Each emits machine-readable JSON for both a clean case and an intentional failure.

| Quickstart      | Clean case | Intentional failure                      |
| --------------- | ---------- | ---------------------------------------- |
| Resume          | `PASS`     | `FAIL` at `scheduler.last_epoch`         |
| Accumulation    | `PASS`     | `FAIL` at `loss_accounting.denominator`  |
| Sample coverage | `PASS`     | `FAIL` at `coverage.same_rank_duplicate` |

No repository checkout is required.

---

## Why TrainParity?

### Checkpoint loading is not the same as resume parity

A checkpoint can load successfully while resumed training differs from uninterrupted training because the state you intended to restore was not preserved in the same way.

That state may include, for example:

* model parameters and buffers;
* optimizer state;
* scheduler state;
* RNG state;
* training position;
* other project-declared checkpoint state.

TrainParity exercises a real process boundary:

```text
uninterrupted execution ─────────────────────────► reference state
                              │
                              └─ save → exit → reload → resumed state
                                                        │
                                                     compare
```

It first runs two uninterrupted baselines to establish self-consistency. If the baseline itself is not reproducible under the declared comparison, TrainParity returns `ABSTAIN` rather than attributing the difference to checkpoint restoration.

---

### Gradient accumulation can silently change an update

A training loop can run normally even when:

```text
microbatch × N
```

does not produce the update you intended to be equivalent to:

```text
one full batch
```

TrainParity compares the declared executions across one optimizer-update boundary:

```text
full batch ───────────────────────────► optimizer update
                                           │
microbatch × N ───────────────────────► optimizer update
                                           │
                                        compare
```

Observed phases include:

* loss accounting;
* gradients;
* optimizer state;
* parameter updates;
* scheduler state.

The user declares the intended relation. TrainParity does not infer that relation from observed values.

---

### A finished epoch does not prove sample coverage

A training epoch can complete without establishing that:

* every expected sample appeared;
* samples appeared exactly once;
* ranks did not overlap;
* declared padding behaved as intended.

TrainParity audits stable sample IDs against one of four explicit policies:

* `exactly_once`
* `at_least_once`
* `no_cross_rank_overlap`
* `expected_padding`

```text
declared expected IDs / policy
              │
              ▼
     observed IDs by rank
              │
              ▼
        coverage audit
```

Coverage claims apply only to the declared finite observation window.

---

## A minimal example

The following `DataLoader` deliberately repeats sample ID `1` and omits sample ID `2`:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from trainparity import ExactlyOnce, Outcome
from trainparity.api import audit_rank_iterables


loader = DataLoader(
    TensorDataset(torch.tensor([0, 1, 1, 3])),
    batch_size=2,
    shuffle=False,
)

result = audit_rank_iterables(
    {0: loader},
    sample_id_extractor=lambda batch: [int(value) for value in batch[0]],
    policy=ExactlyOnce((0, 1, 2, 3)),
)

assert result.outcome is Outcome.FAIL
assert result.first_violation is not None
assert result.first_violation.path == "coverage.same_rank_duplicate"
```

With observed IDs `(0, 1, 2, 3)`, the same declared policy passes.

A complete CI-executed integration is available in
[`examples/test_readme_case.py`](https://github.com/intelland/TrainParity/blob/main/examples/test_readme_case.py).

---

## Resume parity

TrainParity compares uninterrupted execution with save → real process exit → fresh-process load → resumed execution.

A project supplies an importable case describing its command, checkpoint location, and observation semantics. TrainParity supplies the generic process orchestration, baseline self-consistency check, snapshot comparison, and deterministic reporting.

A check can be invoked with:

```python
from trainparity import check_resume

result = check_resume("my_project.trainparity_case:Case")
```

Exact comparison is the default.

If approximate numeric equality is genuinely part of the intended semantics, the tolerance must be supplied explicitly:

```python
from trainparity import ToleranceComparison, check_resume

result = check_resume(
    "my_project.trainparity_case:Case",
    comparison=ToleranceComparison(rtol=1e-6, atol=1e-8),
)
```

TrainParity does not infer or tune tolerance from the observed result.

See the
[external resume integration guide](https://github.com/intelland/TrainParity/blob/main/docs/external-resume-integration.md)
for adapting a real training repository.

---

## Gradient accumulation parity

TrainParity compares a user-declared full-batch execution with an explicitly declared microbatch plan.

A logical comparison covers one optimizer-update boundary.

The baseline and candidate executions begin from verified-equal initial state. If that prerequisite cannot be established, TrainParity returns `ABSTAIN` rather than reporting an accumulation mismatch.

The goal is **not** to assert that every full-batch and microbatch execution should be equivalent. The user supplies the intended equivalence relation; TrainParity tests it.

See the
[public API](https://github.com/intelland/TrainParity/blob/main/docs/api.md)
and the installed accumulation quickstart for the supported interface.

---

## Sample coverage

Coverage checks operate on stable, user-supplied sample IDs.

An ID must be semantically unique inside the declared universe: two different semantic samples must not share the same stable ID.

Exactly-once, at-least-once, and expected-padding policies require a reliable finite expected universe. If that universe is unavailable, the honest result is `ABSTAIN`.

TrainParity validates ID trajectories, not sample contents.

Worker provenance is optional. When worker information is unavailable, it is represented as `None` / JSON `null`, never silently mapped to worker 0.

One audit establishes only the declared policy over one finite observation window.

---

## Validation

TrainParity is validated with controlled faults and pinned external training-code fixtures.

| Surface                      |          Clean controls |                        Deliberate faults / cases | Boundary exercised                        |
| ---------------------------- | ----------------------: | -----------------------------------------------: | ----------------------------------------- |
| Resume reference fixtures    |              6/6 `PASS` | 13/13 detected with the expected first component | Fresh processes; CPU and same-device A100 |
| External resume integrations |              3/3 `PASS` |                                     3/3 detected | Original checkpoint implementations; L40S |
| Accumulation equivalence     |              4/4 `PASS` |                                     8/8 detected | Fresh processes; CPU and same-device L40S |
| Sample coverage              | 17/17 expected outcomes |         Multi-rank and finite-sampler edge cases | World sizes 1/2/3/4                       |

The pinned external resume fixtures use:

* **PyTorch examples ImageNet classifier**
* **nanoGPT**
* **Ignite MNIST Engine recipe**

They exercise the original checkpoint implementations at pinned commits with zero upstream modified LOC.

These are deliberately small product-surface fixtures. The results are **not** a universal bug-detection rate and do not imply compatibility with every use of those projects.

Exact commits, environments, outcomes, artifact identities, and limitations are recorded in
[validation evidence](https://github.com/intelland/TrainParity/blob/main/docs/validation.md).

---

## Design principles

### Explicit semantics

TrainParity does not infer the intended training relation from observed values.

The user supplies the case semantics or coverage policy. Exact comparison is the default for resume and accumulation, and any numeric tolerance must be declared explicitly.

### Real process boundaries

Resume testing includes a real process exit and fresh-process reload.

Accumulation baseline and candidate executions also run in distinct fresh processes.

### Deterministic reporting

Machine reports preserve distinct `PASS`, `FAIL`, `ABSTAIN`, and `ERROR` outcomes and carry both package and report-schema versions.

On `FAIL`, TrainParity identifies the deterministic first observed divergence or policy violation without inferring root cause.

### Conservative conclusions

A passing check establishes only the declared relation or policy over the observed execution.

It does not prove general training correctness, model quality, framework compatibility, or behavior outside the declared observation boundary.

---

## Compatibility

Package metadata permits:

```text
Python >=3.11,<3.12
PyTorch >=2.7,<2.14
```

The `0.1.0` release was explicitly validated on CPython 3.11.15 with CPU PyTorch:

* 2.7.0
* 2.10.0
* 2.13.0

Same-device GPU evidence uses PyTorch 2.7.0 with the exact CUDA and GPU fixtures recorded in
[docs/validation.md](https://github.com/intelland/TrainParity/blob/main/docs/validation.md).

Intermediate PyTorch versions are permitted by the declared dependency range but were not independently validated.

CPU-only users who want to avoid resolving the default CUDA-enabled PyTorch package can install a CPU wheel first, for example:

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install trainparity
```

Untested Python versions, PyTorch versions, operating systems, accelerators, CUDA combinations, distributed configurations, and model scales are outside the recorded validation matrix.

---

## Scope and maturity

TrainParity `0.1.0` is the first non-prerelease release.

The project remains classified as:

```text
Development Status :: 3 - Alpha
```

TrainParity intentionally does **not** claim to:

* detect every training bug;
* infer root causes from observed divergence;
* judge model quality;
* rewrite arbitrary training loops;
* manage checkpoints;
* launch distributed training;
* provide a general Lightning, Transformers, DeepSpeed, DDP, or FSDP adapter layer.

Resume and accumulation execution in the `0.1` contract is single-process training execution with fresh child-process boundaries.

User training code runs with the caller's permissions. TrainParity is not a sandbox. Execute only trusted repositories and load only trusted checkpoints.

See
[Limitations](https://github.com/intelland/TrainParity/blob/main/docs/limitations.md)
and
[Security](https://github.com/intelland/TrainParity/blob/main/SECURITY.md)
for the full boundary.

---

## Documentation

| Topic                       | Link                                                                                                                          |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Public API                  | [docs/api.md](https://github.com/intelland/TrainParity/blob/main/docs/api.md)                                                 |
| External resume integration | [docs/external-resume-integration.md](https://github.com/intelland/TrainParity/blob/main/docs/external-resume-integration.md) |
| Validation evidence         | [docs/validation.md](https://github.com/intelland/TrainParity/blob/main/docs/validation.md)                                   |
| Design                      | [docs/design.md](https://github.com/intelland/TrainParity/blob/main/docs/design.md)                                           |
| Limitations                 | [docs/limitations.md](https://github.com/intelland/TrainParity/blob/main/docs/limitations.md)                                 |
| Security                    | [SECURITY.md](https://github.com/intelland/TrainParity/blob/main/SECURITY.md)                                                 |
| Development provenance      | [docs/development-provenance.md](https://github.com/intelland/TrainParity/blob/main/docs/development-provenance.md)           |
| Comparison with TrainCheck  | [docs/comparison-with-traincheck.md](https://github.com/intelland/TrainParity/blob/main/docs/comparison-with-traincheck.md)   |

---

## Contributing

Bug reports, reproducible parity failures, integration examples, and focused improvements are welcome.

Before proposing a feature outside the frozen `0.1` surface, open an issue describing the observable contract and why the existing public API cannot express it.

See
[CONTRIBUTING.md](https://github.com/intelland/TrainParity/blob/main/CONTRIBUTING.md)
for development and validation requirements.

---

## License

TrainParity is released under the
[MIT License](https://github.com/intelland/TrainParity/blob/main/LICENSE).
