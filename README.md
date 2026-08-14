# TrainParity

[![CI](https://github.com/intelland/TrainParity/actions/workflows/ci.yml/badge.svg)](https://github.com/intelland/TrainParity/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trainparity.svg)](https://pypi.org/project/trainparity/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/intelland/TrainParity/blob/main/LICENSE)

TrainParity checks user-declared equivalence across PyTorch resume, gradient-accumulation, and finite sample-ID executions. It returns `PASS`, `FAIL`, `ABSTAIN`, or `ERROR` and locates the first observed divergence; it is not a universal bug detector and does not invoke an LLM at runtime.

## Install and run

TrainParity is currently alpha-quality software. Version 0.1.0rc5 is a
prerelease of 0.1.0. Install the exact release candidate with:

```bash
pip install trainparity==0.1.0rc5
```

CPU-only users should install a validated PyTorch CPU wheel first so that pip
does not resolve the default CUDA wheel and its runtime packages:

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install trainparity==0.1.0rc5
```

Package metadata still permits `torch>=2.7,<2.14`; the explicitly validated
versions are listed under [Compatibility and security](#compatibility-and-security).

CI executes the pytest integration and all three quickstart commands. The
quickstarts run against the built wheel from outside the repository directory.

## A complete integration

The following compact pytest case audits stable IDs emitted by a real PyTorch
`DataLoader`. The clean loader passes; the faulty loader repeats ID `1` and
omits ID `2`. The file is [`examples/test_readme_case.py`](https://github.com/intelland/TrainParity/blob/main/examples/test_readme_case.py),
and CI executes the command shown below.

```python
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from trainparity import ExactlyOnce, Outcome
from trainparity.api import SampleCoverageResult, audit_rank_iterables


@dataclass(frozen=True)
class CoverageCase:
    sample_ids: tuple[int, ...]
    expected_ids: tuple[int, ...] = (0, 1, 2, 3)

    @staticmethod
    def extract(batch: list[torch.Tensor]) -> list[int]:
        return [int(value) for value in batch[0]]

    def run(self) -> SampleCoverageResult:
        dataset = TensorDataset(torch.tensor(self.sample_ids))
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        return audit_rank_iterables(
            {0: loader},
            sample_id_extractor=self.extract,
            policy=ExactlyOnce(self.expected_ids),
        )


def test_clean_loader_passes() -> None:
    assert CoverageCase((0, 1, 2, 3)).run().outcome is Outcome.PASS


def test_duplicate_loader_reports_first_observed_path() -> None:
    result = CoverageCase((0, 1, 1, 3)).run()
    assert result.outcome is Outcome.FAIL
    assert result.first_violation is not None
    assert result.first_violation.path == "coverage.same_rank_duplicate"
```

From a fresh source checkout, install the development extra before running the
example. It includes `pytest-cov`, which supplies the coverage options used by
the repository-wide pytest configuration:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q --no-cov examples/test_readme_case.py
```

Representative first-observed-divergence output:

```json
{
  "outcome": "FAIL",
  "first_violation": {
    "kind": "same_rank_duplicate",
    "path": "coverage.same_rank_duplicate",
    "sample_id": 1,
    "rank": 0,
    "epoch": 0,
    "position": 2
  },
  "schema_version": 2,
  "trainparity_version": "0.1.0rc5"
}
```

This is the first observed policy violation, not a root-cause claim.

## Installed quickstarts

The installed CPU quickstarts each emit one clean `PASS` and one intentional
`FAIL`:

```bash
python -m trainparity.quickstarts.resume
python -m trainparity.quickstarts.accumulation
python -m trainparity.quickstarts.sample_coverage
```

## Reproducible validation suite

This matrix summarizes the project's pinned, reproducible validation suite. It is not a universal detection rate and does not establish correctness for untested projects, devices, or training semantics.

| Surface | Clean controls | Deliberate faults / cases | Boundary exercised |
| --- | ---: | ---: | --- |
| Resume reference fixtures | 6/6 PASS | 13/13 detected with expected first component | Fresh process; CPU and same-device A100 |
| External resume integrations | 3/3 PASS | 3/3 detected | Original checkpoint paths; ImageNet, nanoGPT, Ignite; L40S |
| Accumulation equivalence | 3 CPU + 1 GPU PASS | 8/8 detected | Fresh processes; explicit phases; same-device L40S |
| Sample coverage | 17/17 expected outcomes | world sizes 1/2/3/4 and finite sampler edge cases | CPU; finite declared windows |

Exact commits, environments, outcomes, and limitations are in [validation](https://github.com/intelland/TrainParity/blob/main/docs/validation.md). The external-project results used tiny fixtures; they are evidence about those cases, not a framework-compatibility promise.

## What it checks

- Resume equivalence compares a continuous execution with save, real exit, fresh-process load, and resumed execution through the project's checkpoint semantics.
- Accumulation equivalence compares a declared full-batch execution with a declared microbatch plan at bounded loss-accounting, gradient, optimizer/parameter, and scheduler phases.
- Sample coverage evaluates only explicit `exactly_once`, `at_least_once`, `no_cross_rank_overlap`, or `expected_padding` policies over stable sample IDs in a finite observation window.

The four outcomes are intentionally distinct:

- `PASS`: the declared observations satisfied the comparison or policy.
- `FAIL`: an observed difference or policy violation was found.
- `ABSTAIN`: required evidence was unavailable or ambiguous, such as an unknown expected universe for exactly-once coverage.
- `ERROR`: execution or observation could not complete.

Resume and accumulation checks default to `ExactComparison`. Both accept an
explicit user-configured `ToleranceComparison`; TrainParity does not infer or
tune a tolerance from observed results. Sample coverage instead uses its
declared discrete coverage policy and is not a numeric comparison.

For example, a resume check may declare its numeric relation explicitly:

```python
from trainparity import ToleranceComparison, check_resume

result = check_resume(
    "trainparity_case:Case",
    comparison=ToleranceComparison(rtol=1e-6, atol=1e-8),
)
```

This tolerance is user-declared semantics, not TrainParity deciding that an
observed difference is small enough.

## User contract

Resume and accumulation checks require an importable case that states project semantics: how to execute, locate/load a checkpoint or construct one optimizer-update boundary, and expose the required state. External resume checks can also require launcher or checkpoint-location glue when the upstream interface is implicit or timestamped. TrainParity owns generic fresh-process orchestration and deterministic reporting; it does not rewrite a training loop or provide framework-specific adapters. See the [external resume integration guide](https://github.com/intelland/TrainParity/blob/main/docs/external-resume-integration.md), [public API](https://github.com/intelland/TrainParity/blob/main/docs/api.md), [design](https://github.com/intelland/TrainParity/blob/main/docs/design.md), and shipped quickstart modules.

Coverage users provide stable sample IDs. An ID must be semantically unique within the declared expected universe: two different semantic samples must not share it. TrainParity validates ID trajectories, not sample contents. Worker provenance is optional and unavailable worker information is represented by `None` / JSON `null`, never worker 0. One audit proves only one finite observation window—the declared window; it does not prove sample contents, infinite-stream exactly-once behavior, or general shuffle quality.

## What it does not do

TrainParity does not diagnose arbitrary scripts, infer root causes, judge model quality, launch distributed jobs, manage checkpoints, or provide Lightning, Transformers, DeepSpeed, DDP, FSDP, dashboard, service, registry, or runtime agent integration. It does not claim that all full-batch and microbatch executions should be equivalent; the user declares the relation and any tolerance.

Implementation provenance and the separation between assisted development and
deterministic runtime behavior are documented in [development provenance](https://github.com/intelland/TrainParity/blob/main/docs/development-provenance.md).

[TrainCheck](https://github.com/OrderLab/TrainCheck) infers and checks training invariants using reference and target traces. TrainParity performs explicit A/B differential tests over user-declared equivalence relations and fresh-process boundaries. Neither structural approach makes the other a universal detector. The scoped comparison and cited upstream material are in [comparison with TrainCheck](https://github.com/intelland/TrainParity/blob/main/docs/comparison-with-traincheck.md).

## Compatibility and security

Package metadata permits `torch>=2.7,<2.14`. The release validation matrix
explicitly tested the installed CPU wheel on CPython 3.11 with
PyTorch 2.7.0, 2.10.0, and 2.13.0. Intermediate PyTorch versions in the
declared range are installable but were not independently validated. Same-device GPU evidence uses
PyTorch 2.7.0 on the exact CUDA/GPU fixtures listed in
[validation](https://github.com/intelland/TrainParity/blob/main/docs/validation.md). No support is implied outside this declared and tested scope.

TrainParity is not a sandbox. User training code runs with the caller's permissions; load only trusted checkpoints and do not execute untrusted repositories. Explicit child environment values are propagated when requested but are not recorded in reports by default. See [SECURITY.md](https://github.com/intelland/TrainParity/blob/main/SECURITY.md).

Known constraints and non-claims are collected in [limitations](https://github.com/intelland/TrainParity/blob/main/docs/limitations.md). Contributions should follow [CONTRIBUTING.md](https://github.com/intelland/TrainParity/blob/main/CONTRIBUTING.md). This project is MIT licensed.
