# Gate 2 snapshot and comparison contract

## Schema and backend boundary

`Snapshot` records a logical step, schema version, backend name, and immutable
state tree. Gate 2 ships `FullValueBackend` as the correctness reference: tensor
values are copied into immutable bytes while original shape, dtype, device, and
`requires_grad` metadata are retained. Mutating the model, optimizer, extras,
or a materialized comparison tensor cannot change an existing snapshot.

The capture entry point accepts the `SnapshotBackend` protocol. Full-value
materialization is therefore the Gate 2 reference behavior, not a promise that
all future backends must store full tensor bytes.

## Stable paths

Mappings are traversed in lexical key order; lists and tuples preserve order
and kind. PyTorch parameter names are expanded on `.` boundaries so optimizer
paths are readable and stable:

```text
model.encoder.weight
gradient.encoder.weight
optimizer.param_groups[0].lr
optimizer.state.encoder.weight.exp_avg
extra.ema.shadow.weight
```

Non-identifier user mapping keys use JSON bracket quoting, so `"a.b"` cannot
collide with nested `a -> b`. Missing, explicit `None`, empty containers/tensors,
zero, NaN, and Inf remain distinct states.

## Optimizer canonicalization

The canonicalizer maps live optimizer parameters through
`model.named_parameters(remove_duplicate=False)`. It preserves parameter-group
and within-group order but replaces parameter objects and serialized IDs with
names. Momentum, Adam, and empty parameter states are nested under those names.

Capture returns `ABSTAIN` with a precise path when a parameter has no model
name, has aliases, occurs more than once, or an optimizer state key cannot be
mapped uniquely. It never falls back to memory or checkpoint IDs.

## Comparison policies

`ExactComparison` requires identical structure, metadata, scalar float bits,
and tensor bytes. This deliberately distinguishes signed zero; identical NaN
payloads compare exactly.

`ToleranceComparison(rtol, atol, equal_nan)` is a separate class. Users must
provide finite, non-negative tolerances. It applies numerical tolerance only
after structure, sequence kind, shape, dtype, device, and `requires_grad` match;
non-numerical and integral state remains exact. It never guesses or widens a
tolerance.

A failure contains path, reason, compact baseline/candidate summaries, and
numerical error where meaningful. It is described only as the **first observed
divergence**, never as root cause.

## Gate boundary

Gate 2 captures and compares one snapshot at a time. It does not launch child
processes, save/reload training, compare trajectories, or implement resume,
accumulation, or sample-coverage orchestration.

