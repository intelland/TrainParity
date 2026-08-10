# TrainParity Status

## Active gate

Gate 0 — ecosystem and differentiation validation.

## Objective

Produce a reproducible competitor study against OrderLab/TrainCheck and a
precise product contract for TrainParity. Validate the structural value of an
explicit A/B execution model with first-observed-divergence reporting before
building the installable library.

## Constraints

- Implement Gate 0 only and stop for human acceptance.
- Run Python, PyTorch, competitor, and experiment workloads on M3, not locally.
- Keep every environment, cache, checkout, log, and output under
  `/scratch/mp25/jwuu0254/zxh/TrainParity`.
- Do not copy OrderLab/TrainCheck code.
- Keep the throwaway TrainParity prototype below 100 logical lines.
- Preserve four stable faults: missing scheduler state, missing RNG state,
  gradient-accumulation mean-of-means, and sample duplication.
- Conclude `STOP` if the differentiation is not structural.

## Planned verification commands

Run from the M3 repository checkout unless noted otherwise:

```bash
python -m experiments.gate0.run_fault_matrix \
  --output "$PROJECT_ROOT/outputs/gate0/recorded/fault_matrix.json"
python scripts/verify_gate.py 0
git diff --check
```

The final Gate 0 report will record the exact environment, commands, outcomes,
and any blocked competitor experiments.

## Current state

Gate 0 started. Handoff specifications have been imported; implementation and
competitor experiments are pending.
