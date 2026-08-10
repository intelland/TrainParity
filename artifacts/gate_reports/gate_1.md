# Gate 1 report

## Outcome

**PASS — recommendation: HUMAN_REVIEW**

The installable Gate 1 skeleton and class/protocol adapter satisfy machine acceptance. Human approval is required before Gate 2.

## Acceptance criteria

- [x] installable skeleton and engineering files: present
- [x] two API prototypes evaluated: class/protocol selected after factory-plus-callback comparison
- [x] selected adapter size: logical_lines=28
- [x] fresh-process import: returncode=0
- [x] no cloudpickle requirement: both prototypes use ordinary module imports
- [x] correct resume case: outcome=PASS
- [x] faulty resume case: outcome=FAIL, first observed=optimizer.param_groups.0.lr
- [x] minimal documented production dependencies: dependencies=['torch>=2.5']
- [x] wheel build: wheels=['trainparity-0.1.0.dev1-py3-none-any.whl']
- [x] CI covers Gate 1 checks: lint, type-check, pytest, wheel, and verifier steps present
- [x] accepted Gate 0 evidence preserved: hashes unchanged

## API selection

Selected: `class_protocol`.

Selected simple adapter: 28 logical lines.
Correct resume example: `PASS`.
Faulty resume example: `FAIL` at
`optimizer.param_groups.0.lr`.

## Exact commands

- `python -m experiments.gate1.run_adapter_evaluation --output $PROJECT_ROOT/outputs/gate1/adapter_evaluation.json`
- `make lint`
- `make typecheck`
- `make test`
- `make build`
- `python scripts/verify_gate.py 1`
- `git diff --check`

## Remaining limitations

- The direct resume probe is Gate 1 evidence, not a production runner or comparator.
- The tiny CPU cases do not establish compatibility with real training repositories.
- Only ordinary importable zero-argument classes are accepted; arbitrary scripts and local closures are unsupported.
- Distributed training, framework adapters, services, and runtime LLM/agent integration are out of scope.
- First observed divergence is not presented as root cause.
