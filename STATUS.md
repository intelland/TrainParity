# TrainParity Status

## Active gate

Gate 2 — deterministic snapshot, canonicalization, and comparison core.

## Objective

Implement a deterministic full-value reference snapshot backend, stable state
paths, name-based optimizer canonicalization, and separate exact/tolerance
comparators that report first observed differences without claiming root cause.

## Constraints

- Implement Gate 2 only and stop for human acceptance.
- Run Python, PyTorch, competitor, and experiment workloads on M3, not locally.
- Keep every environment, cache, checkout, log, and output under
  `/scratch/mp25/jwuu0254/zxh/TrainParity`.
- Preserve all accepted Gate 0 evidence unchanged.
- Preserve all accepted Gate 0 and Gate 1 evidence unchanged.
- Define deterministic, unambiguous state paths and freeze captured tensor
  values without retaining aliases to mutable training state.
- Distinguish missing, `None`, empty, zero, NaN, and Inf states.
- Keep `ExactComparison` and explicit `ToleranceComparison` separate.
- Canonicalize optimizer state by stable parameter names and return `ABSTAIN`
  when mapping is missing, duplicated, aliased, or otherwise ambiguous.
- Implement a full-value reference backend without making materialized
  snapshots the only possible future backend.
- Do not add runtime LLM/agent dependencies, distributed support, a web UI,
  service, registry, or platform functionality.
- Do not implement production resume orchestration, phase tracing, accumulation,
  sample coverage, or any other later-gate feature.
- Continue to describe outputs as first observed divergence, never root cause.

## Planned verification commands

Run from the M3 repository checkout unless noted otherwise:

```bash
make lint
make typecheck
make test
make build
python scripts/verify_gate.py 2
git diff --check
```

The final Gate 2 report will record the fault/clean suites, stable expected
paths, coverage, optimizer ambiguity behavior, exact/tolerance behavior,
environment, exact command outcomes, and all known limitations.

## Current state

Gate 0 was accepted by the human reviewer on 2026-08-10. Its machine report is
`PASS` with recommendation `GO`; the report and all supporting evidence remain
preserved in their existing paths.

Gate 1 was accepted by the human reviewer on 2026-08-10. Its 28-line selected
adapter, wheel-installed fresh-process import, clean `PASS`, and faulty `FAIL`
evidence remain preserved.

Gate 2 is authorized and in progress. The hosted GitHub Actions result for
commit `ae75212` was not independently verified because the private repository
is inaccessible through the connected GitHub App and no authenticated GitHub
CLI is available. The equivalent full M3 sequence passed and was accepted for
Gate 1. Confirming the remote Actions result is a carry-forward requirement
that must be resolved or precisely reported no later than Gate 2 review.

No Gate 3 work is authorized.
