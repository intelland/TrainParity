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

Gate 2 is complete and awaiting human acceptance. The hosted GitHub Actions carry-forward
is resolved: an authenticated read-only REST query confirmed that commit
`ae75212` completed `CI` run `31394676144` with conclusion `success`. The query
used the local Git credential helper without printing or persisting credentials;
the durable record is `experiments/gate2/recorded/ci_ae75212.json`.

The final Gate 2 M3 evidence has 17/17 expected fault paths, 0/17 clean false
positives, an optimizer-alias `ABSTAIN`, and 96.25% core coverage across 51
passing tests. Ruff, strict Mypy across 13 package modules, wheel/sdist build,
fault evidence replay, `git diff --check`, and `python scripts/verify_gate.py 2`
all pass in the isolated CPU environment at `envs/gate2`. The verifier reports
`PASS` with recommendation `HUMAN_REVIEW`; see
`artifacts/gate_reports/gate_2.json` and `gate_2.md`.

No Gate 3 work is authorized.
