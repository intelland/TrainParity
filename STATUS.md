# TrainParity Status

## Active gate

Gate 1 — product contract, API prototypes, and engineering skeleton.

## Objective

Create an installable Python package skeleton and compare two explicit adapter
APIs using one correct and one faulty resume example. Select the lower-friction,
process-safe design without implementing the Gate 2 snapshot/comparator or the
Gate 3 resume-equivalence runner.

## Constraints

- Implement Gate 1 only and stop for human acceptance.
- Run Python, PyTorch, competitor, and experiment workloads on M3, not locally.
- Keep every environment, cache, checkout, log, and output under
  `/scratch/mp25/jwuu0254/zxh/TrainParity`.
- Preserve all accepted Gate 0 evidence unchanged.
- Compare class/protocol and factory-plus-callback adapter forms.
- Keep the selected simple adapter at or below 30 logical lines and importable
  in a fresh Python process without `cloudpickle`.
- Keep production dependencies minimal and document each one.
- Do not add runtime LLM/agent dependencies, distributed support, a web UI,
  service, registry, or platform functionality.
- Do not implement later-gate snapshot/comparison or orchestration features.

## Planned verification commands

Run from the M3 repository checkout unless noted otherwise:

```bash
make lint
make typecheck
make test
make build
python scripts/verify_gate.py 1
git diff --check
```

The final Gate 1 report will record exact environment and command outcomes,
the measured adapter line count, fresh-process import evidence, API selection,
and all known limitations.

## Current state

Gate 0 was accepted by the human reviewer on 2026-08-10. Its machine report is
`PASS` with recommendation `GO`; the report and all supporting evidence remain
preserved in their existing paths.

Gate 1 implementation and M3 verification are complete and awaiting human
acceptance. The selected class/protocol adapter is 28 logical lines, imports
from a fresh process and an installed wheel, and requires no `cloudpickle`.
The correct resume example returns `PASS`; the deliberately missing scheduler
state returns `FAIL` with first observed divergence at
`optimizer.param_groups.0.lr`. This observation is not presented as root cause.

On the isolated M3 CPU environment, Ruff passed, Mypy passed for all eight
package modules, pytest passed 10 tests, the wheel and source distribution
built successfully, and `python scripts/verify_gate.py 1` returned `PASS` with
recommendation `HUMAN_REVIEW`. The accepted Gate 0 evidence hashes remain
unchanged. See `artifacts/gate_reports/gate_1.json` and `gate_1.md`.

The GitHub Actions workflow runs the same lint, type-check, test, build, and
verifier sequence. An intermediate run exposed the Ruff configuration scope;
that issue was corrected and the identical M3 command now passes. Gate 2 has
not started and remains unauthorized pending explicit human approval.
