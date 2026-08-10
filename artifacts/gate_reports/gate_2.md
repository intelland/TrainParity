# Gate 2 report

## Outcome

**PASS — recommendation: HUMAN_REVIEW**

Gate 2 snapshot, canonicalization, and comparison satisfy machine acceptance. Human approval is required before Gate 3.

## Acceptance criteria

- [x] Gate 2 implementation and evidence files: present
- [x] required fault inventory: observed=['adam_exp_avg', 'device_metadata', 'empty_tensor', 'extra_key', 'inf', 'missing_key', 'missing_vs_none', 'nan', 'nested_value', 'none_vs_zero', 'parameter_group_order', 'same_value_different_path', 'sgd_momentum', 'tensor_dtype', 'tensor_shape', 'tensor_value', 'user_state_dict']
- [x] fault suite expected stable paths: 17/17 exact paths
- [x] clean suite zero false positives: 0/17
- [x] actionable non-binary difference reports: path, reason, baseline, and candidate present for every fault
- [x] exact and tolerance policies remain separate: observed={'exact': 'FAIL', 'tolerance': 'PASS'}
- [x] ambiguous optimizer mapping abstains: outcome=ABSTAIN, path=optimizer.param_groups[0].params[0]
- [x] optimizer paths use names: paths=['optimizer.state.weight.exp_avg', 'optimizer.state.weight.momentum_buffer']
- [x] captured tensor state is alias-free: mutation probe=PASS
- [x] core module coverage: percent_covered=96.25
- [x] accepted Gate 0 and Gate 1 evidence preserved: hashes unchanged
- [x] CI runs Gate 2 verifier: lint, type-check, tests, build, and Gate 2 verifier configured
- [x] Gate 1 hosted CI carry-forward confirmed: source=authenticated read-only GitHub REST query through the local Git credential helper, conclusion=success

## Representative first observed differences

- `tensor_shape` at `extra.tensor` (tensor_shape): (2,) -> (3,)
- `nan` at `extra` (value): nan -> 0.0
- `parameter_group_order` at `optimizer.param_groups[0].lr` (value): 0.1 -> 0.2
- `sgd_momentum` at `optimizer.state.weight.momentum_buffer` (tensor_value): tensor(shape=(1,), dtype=torch.float32, device=cpu, values=[1.0]) -> tensor(shape=(1,), dtype=torch.float32, device=cpu, values=[2.0])
- `user_state_dict` at `extra.ema.shadow.weight` (value): 1 -> 2

These are first observed divergences, not root-cause claims.

## Metrics

- Fault paths matched: 17/17
- Clean false positives: 0
- Core coverage: 96.25%
- Ambiguous optimizer outcome: `ABSTAIN`

## Exact commands

- `python -m experiments.gate2.run_fault_suite --output $PROJECT_ROOT/outputs/gate2/fault_suite.json`
- `make lint`
- `make typecheck`
- `make test`
- `make build`
- `python scripts/verify_gate.py 2`
- `git diff --check`

## Remaining limitations

- FullValueBackend is a correctness reference, not the only permitted future storage backend.
- Gate 2 compares one snapshot and does not implement trajectory or resume orchestration.
- Sparse and non-strided tensors currently produce ABSTAIN.
- CUDA metadata behavior is contract-tested without requiring a GPU allocation.
- First observed divergence is not presented as root cause.
