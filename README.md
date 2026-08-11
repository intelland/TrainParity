# TrainParity

[![CI](https://github.com/intelland/TrainParity/actions/workflows/ci.yml/badge.svg)](https://github.com/intelland/TrainParity/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trainparity.svg)](https://pypi.org/project/trainparity/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

TrainParity is a deterministic PyTorch differential-testing library that checks user-declared equivalence across resume, gradient-accumulation, and finite sample-ID trajectories and reports their first observed divergence.

A checkpoint that restores model weights but resets the scheduler can look healthy while changing every later update. Run the installed CPU resume example:

```bash
python -m trainparity.quickstarts.resume
```

It performs a clean check and the intentional scheduler-reset fault, producing a report shaped like:

```json
{
  "clean": {"outcome": "PASS"},
  "intentional_fail": {
    "outcome": "FAIL",
    "primary_difference": {
      "path": "scheduler.last_epoch",
      "baseline": 4,
      "candidate": 2
    }
  },
  "schema_version": 1,
  "trainparity_version": "0.1.0rc1"
}
```

This is a first observed divergence, not a root-cause claim. TrainParity is not a universal bug detector and does not invoke an LLM at runtime.

## Reproducible validation suite

This matrix summarizes the project's pinned, reproducible validation suite. It is not a universal detection rate and does not establish correctness for untested projects, devices, or training semantics.

| Surface | Clean controls | Deliberate faults / cases | Boundary exercised |
| --- | ---: | ---: | --- |
| Resume reference fixtures | 6/6 PASS | 13/13 detected with expected first component | Fresh process; CPU and same-device A100 |
| External resume integrations | 3/3 PASS | 3/3 detected | Original checkpoint paths; ImageNet, nanoGPT, Ignite; L40S |
| Accumulation equivalence | 3 CPU + 1 GPU PASS | 8/8 detected | Fresh processes; explicit phases; same-device L40S |
| Sample coverage | 17/17 expected outcomes | world sizes 1/2/3/4 and finite sampler edge cases | CPU; finite declared windows |

Exact commits, environments, outcomes, and limitations are in [validation](docs/validation.md). The external-project results used tiny fixtures; they are evidence about those cases, not a framework-compatibility promise.

## Install and run

The project is an unpublished release candidate. Build the wheel from this checkout and install that artifact; do not interpret the PyPI badge as a publication claim.

The three installed, CPU-runnable examples each emit a clean `PASS` and a small intentional `FAIL`:

```bash
python -m trainparity.quickstarts.resume
```

```bash
python -m trainparity.quickstarts.accumulation
```

```bash
python -m trainparity.quickstarts.sample_coverage
```

CI executes these exact commands against the built wheel from outside the repository directory.

## What it checks

- Resume equivalence compares a continuous execution with save, real exit, fresh-process load, and resumed execution through the project's checkpoint semantics.
- Accumulation equivalence compares a declared full-batch execution with a declared microbatch plan at bounded loss-accounting, gradient, optimizer/parameter, and scheduler phases.
- Sample coverage evaluates only explicit `exactly_once`, `at_least_once`, `no_cross_rank_overlap`, or `expected_padding` policies over stable sample IDs in a finite observation window.

The four outcomes are intentionally distinct:

- `PASS`: the declared observations satisfied the comparison or policy.
- `FAIL`: an observed difference or policy violation was found.
- `ABSTAIN`: required evidence was unavailable or ambiguous, such as an unknown expected universe for exactly-once coverage.
- `ERROR`: execution or observation could not complete.

`ExactComparison` and user-configured `ToleranceComparison` remain separate. TrainParity does not infer or tune a tolerance from observed results.

## User contract

Resume and accumulation checks require a small importable case that states project semantics: how to execute, locate/load a checkpoint or construct one optimizer-update boundary, and expose the required state. TrainParity owns generic fresh-process orchestration and deterministic reporting; it does not rewrite a training loop or provide framework-specific adapters. See the [public API](docs/api.md), [design](docs/design.md), and shipped quickstart modules.

Coverage users provide stable sample IDs. An ID must be semantically unique within the declared expected universe: two different semantic samples must not share it. TrainParity validates ID trajectories, not sample contents. Worker provenance is optional and unavailable worker information is represented by `None` / JSON `null`, never worker 0. One audit proves only one finite observation window—the declared window; it does not prove sample contents, infinite-stream exactly-once behavior, or general shuffle quality.

## What it does not do

TrainParity does not diagnose arbitrary scripts, infer root causes, judge model quality, launch distributed jobs, manage checkpoints, or provide Lightning, Transformers, DeepSpeed, DDP, FSDP, dashboard, service, registry, or runtime agent integration. It does not claim that all full-batch and microbatch executions should be equivalent; the user declares the relation and any tolerance.

Asking Codex can help inspect or modify code, but it is an interactive, probabilistic development activity. TrainParity instead runs fixed local code, declared observations, explicit predicates, and versioned machine reports. Codex assisted this repository's implementation under human-defined gated specifications; it is absent from runtime decisions. See [development provenance](docs/development-provenance.md).

[TrainCheck](https://github.com/OrderLab/TrainCheck) infers and checks training invariants using reference and target traces. TrainParity performs explicit A/B differential tests over user-declared equivalence relations and fresh-process boundaries. Neither structural approach makes the other a universal detector. The scoped comparison and cited upstream material are in [comparison with TrainCheck](docs/comparison-with-traincheck.md).

## Compatibility and security

The release candidate is tested on CPython 3.11 and PyTorch 2.7.0, with CPU plus the exact CUDA/GPU fixtures listed in [validation](docs/validation.md). No support is implied for other Python, PyTorch, CUDA, or GPU versions.

TrainParity is not a sandbox. User training code runs with the caller's permissions; load only trusted checkpoints and do not execute untrusted repositories. Explicit child environment values are propagated when requested but are not recorded in reports by default. See [SECURITY.md](SECURITY.md).

Known constraints and non-claims are collected in [limitations](docs/limitations.md). Contributions should follow [CONTRIBUTING.md](CONTRIBUTING.md). This project is MIT licensed.
