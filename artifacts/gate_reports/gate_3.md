# Gate 3 report

## Outcome

**PASS — recommendation: GO**

Gate 3 proves the tiny reference cases across real process and same-device GPU boundaries. Human review is required before Gate 4.

## Acceptance criteria

- [x] Gate 3 implementation and evidence files: present
- [x] clean fixtures have zero false positives: three CPU and three same-device GPU clean runs passed
- [x] formal stable fault suite detects every fault: detected=13/13
- [x] expected first component threshold: matched=13/13
- [x] real process boundary: all recorded pre-save and post-load PIDs are distinct
- [x] strict ABSTAIN and ERROR controls: baseline nondeterminism=ABSTAIN, child exception=ERROR
- [x] formal aligned step and data semantics: step N follows N updates; cursor fault first observed at batch.sample_ids
- [x] single real GPU same-device matrix: gpu=NVIDIA A100 80GB PCIe, job=58957857
- [x] unit, contract, and integration test coverage: tests=71 passed, coverage=94.86%
- [x] accepted Gate 0-2 evidence preserved: hashes unchanged
- [x] CI runs Gate 3 verifier: lint, type-check, tests, build, and Gate 3 verifier configured

## First observed fault divergences

- `missing_model`: step 2, `model.bias`
- `missing_optimizer`: step 2, `optimizer.param_groups[0].lr`
- `missing_scheduler`: step 2, `scheduler._last_lr[0]`
- `missing_python_rng`: step 2, `rng.python[1][0]`
- `missing_numpy_rng`: step 2, `rng.numpy.keys.values[0]`
- `missing_torch_cpu_rng`: step 2, `rng.torch_cpu`
- `data_cursor_offset`: step 3, `batch.sample_ids[0]`
- `resume_step_off_by_one`: step 2, `step`
- `optimizer_parameter_group_mismatch`: step 2, `optimizer.param_groups[0].lr`
- `extra_scheduler_step`: step 2, `scheduler._step_count`
- `missing_hidden_module_global`: step 2, `extra.hidden_module_counter`
- `missing_cuda_rng`: step 2, `rng.torch_cuda.device_0`
- `missing_grad_scaler`: step 2, `scaler.scale`

These are first observed divergences, not root-cause claims. Every difference
at the first divergent step remains available in the raw M3 matrix outputs.

## Metrics

- Clean false positives: 0
- Stable faults detected: 13/13
- Expected first component: 13/13
- CPU repeats: 3
- GPU repeats: 3
- GPU: NVIDIA A100 80GB PCIe (Slurm job 58957857)
- Tests / coverage: 71 passed / 94.86%

## Exact commands

- `make lint`
- `make typecheck`
- `make test`
- `make build`
- `python -m experiments.gate3.run_cpu_matrix --output $PROJECT_ROOT/outputs/gate3/cpu_matrix.json`
- `sbatch scripts/slurm_gpu_matrix.sbatch --gate 3`
- `python scripts/verify_gate.py 3`
- `git diff --check`

## Remaining limitations

- Only tiny single-process cases and one A100 were evaluated; real-project friction belongs to Gate 4.
- Only the completed-training-step phase is supported; accumulation and phase tracing are not implemented.
- The full-value snapshot backend prioritizes correctness and has not been performance-optimized.
- Stable sample identity is required; missing identity returns ABSTAIN.
- Exact comparison is used and no numeric tolerance is inferred.
- First observed divergence is not presented as root cause.
