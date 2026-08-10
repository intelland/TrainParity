# Gate 0 report

## Outcome

**PASS — recommendation: GO**

Controlled evidence supports an explicit A/B plus first-state-path product distinction. Human approval is required before Gate 1.

## Acceptance criteria

- [x] required artifacts: present
- [x] fault fixture inventory: observed=['mean_of_means', 'missing_rng_state', 'missing_scheduler_state', 'sample_duplication']
- [x] stable expected first divergences: all four exact
- [x] three-run repeatability: stable=True repeats=3
- [x] prototype size: physical=79, logical=64
- [x] competitor inventory: observed=['mean_of_means', 'missing_rng_state', 'missing_scheduler_state', 'sample_duplication']
- [x] TrainCheck black-box execution: 24/24 phases exited zero
- [x] clean-control correction: control and fault signatures recorded
- [x] structural differentiation threshold: prototype-only precise detections=['mean_of_means', 'missing_rng_state', 'sample_duplication']
- [x] competitor sources and limitations: official sources, controls, and threats documented
- [x] four-state product contract: PASS/FAIL/ABSTAIN/ERROR and first-observed semantics documented

## TrainParity prototype

- `missing_scheduler_state`: step 2, `optimizer.lr`
- `missing_rng_state`: step 2, `rng.torch`
- `mean_of_means`: step 0, `gradient.model.weight`
- `sample_duplication`: step 3, `batch.sample_ids.0`

Prototype size: 79 physical lines,
64 nonblank/noncomment lines.

## TrainCheck 0.1.2 with clean controls

- `missing_scheduler_state`: control=14, fault=27, specific=13, detected=True
- `missing_rng_state`: control=6, fault=6, specific=0, detected=False
- `mean_of_means`: control=5, fault=5, specific=0, detected=False
- `sample_duplication`: control=6, fault=6, specific=0, detected=False

## Exact commands

- `python -m experiments.gate0.run_fault_matrix --output $PROJECT_ROOT/outputs/gate0/recorded/fault_matrix.json`
- `python -m experiments.gate0.run_traincheck_matrix --runtime-root $PROJECT_ROOT/outputs/gate0/traincheck --output $PROJECT_ROOT/outputs/gate0/recorded/traincheck_summary.json`
- `python -m ruff check experiments/gate0 scripts/verify_gate.py`
- `python -m mypy --ignore-missing-imports --check-untyped-defs experiments/gate0 scripts/verify_gate.py`
- `python -m compileall -q experiments scripts`
- `python scripts/verify_gate.py 0`

## Remaining limitations

- Four tiny CPU fixtures do not establish real-repository adapter cost.
- TrainCheck was tested only at version 0.1.2 with PyTorch 2.13.0+cpu and the pandas backend.
- The throwaway prototype does not implement a true fresh-process resume runner.
- Gate 0 provides no production TrainParity API or arbitrary-script support.
