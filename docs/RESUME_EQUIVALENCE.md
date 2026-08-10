# Resume equivalence contract

Gate 3 compares a continuous execution with an interrupted execution that
saves an actual checkpoint file, terminates, starts a fresh Python interpreter,
constructs new objects, loads the checkpoint, and resumes.

## Step and phase semantics

Snapshot `step=N` means exactly `N` optimizer updates have completed. Gate 3
has one phase, `completed_training_step`. Step zero is the initial state before
any optimizer update. The resumed snapshot at the split boundary comes from the
fresh load process, not the process that saved the checkpoint.

The runner compares two independently executed continuous baselines first. A
difference between those baselines returns `ABSTAIN`, because resume attribution
would be unsafe. It also compares independently built initial snapshots before
attributing a difference to checkpoint/resume behavior.

## Adapter observation

The Gate 1 adapter gains one method:

```python
def observe(self, state: TrainingState) -> StepObservation: ...
```

After each completed step, `StepObservation` must provide either stable sample
IDs or a deterministic batch fingerprint. Missing stable data identity returns
`ABSTAIN`. Optional `extras` capture user state such as a module-global counter.
Captured values are frozen by the full-value reference backend.

## Outcomes

- `PASS`: every aligned snapshot is equivalent under exact comparison.
- `FAIL`: a controlled trajectory has a first observed divergence.
- `ABSTAIN`: prerequisites such as deterministic baselines or stable batch
  identity are not satisfied.
- `ERROR`: import, worker launch, timeout, result corruption, checkpoint I/O,
  load, disk, or child execution failed.

A failure contains the last matching step, first divergent step, phase,
deterministic primary difference, and every other difference at that same step.
These are observations, not root-cause claims.

## Process and GPU boundaries

Workers are invoked through ordinary `python -m trainparity.worker` imports;
cloudpickle is not used. Pre-save and post-load PIDs plus model, optimizer,
scheduler, and scaler object identities are recorded. The pre-save worker fully
exits before the post-load worker starts.

CUDA validation runs all continuous and resumed workers inside one Slurm job
with exactly one visible GPU. It tests a clean case, omitted CUDA RNG, and
omitted GradScaler state. Results from different GPU models are never compared.

## Deliberate limits

Gate 3 does not implement distributed training, accumulation or phase tracing,
snapshot performance optimization, full sample-coverage policy, framework
adapters, a service, or automated diagnosis. Exact comparison is the runner
policy; no numeric tolerance is guessed.
