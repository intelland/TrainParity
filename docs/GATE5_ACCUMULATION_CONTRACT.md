# Gate 5 Accumulation Equivalence Contract

Gate 5 tests a **user-declared** equivalence relation between one full-batch
execution and one explicitly planned microbatch execution. TrainParity does not
claim that every full-batch and accumulated execution is equivalent. In
particular, batch-dependent layers, stochastic operations, reduction order, and
stateful forward passes can make the declaration false or unsupported.

## One optimizer-update boundary

One logical update starts from a verified-equal freshly constructed state. It
consumes exactly the samples in one declared full batch, performs the declared
loss reduction and backward operations, performs one expected
`optimizer.step()`, and then performs the declared scheduler action. The
boundary ends after that scheduler action. A faulty plan may perform an action
more or less often inside this window; that is an observed mismatch, not a
different definition of a step.

Baseline A and baseline B execute the full-batch plan in separate fresh
processes. Their initial states and bounded observations must match before the
candidate can be attributed. The candidate executes the microbatch plan in a
third fresh process from the same seed and its initial state must match the
baseline initial state.

## Bounded observations

The runner records only these ordered observations:

1. `loss_accounting`: the effective loss used for backward and, when supplied,
   its numerator and denominator;
2. `gradient`: gradients after the complete backward phase and before the
   expected optimizer step;
3. `optimizer_state`: optimizer state after the update;
4. `parameter_update`: model parameters and buffers after the update;
5. `scheduler_state`: scheduler and scaler state at the boundary.

The earliest mismatch is a **first observed divergence**, never a root-cause
claim. This is a bounded accumulation check, not a general event tracer.

## Batch and loss contracts

A case may provide an explicit splitter. Otherwise the default splitter accepts
only a tensor tree made of tensors, string-keyed mappings, tuples, and lists. It
splits every tensor along dimension zero, requires equal leading lengths, keeps
leaf structure and sample order, and rejects empty, scalar, inconsistent, or
unsupported leaves with `ABSTAIN`. It never silently changes samples, masks,
labels, or weights.

Loss accounting is optionally explicit as numerator plus denominator. When all
microbatches provide it, the runner uses the sum of numerators divided by the
sum of denominators. When it is unavailable, the report says
`loss_normalization_captured=false`; no denominator is inferred. Mixed or
invalid accounting abstains.

Exact comparison and explicitly configured tolerance comparison are separate
policies. TrainParity never derives or tunes a tolerance from observations.

## Known non-equivalence and scope

BatchNorm training statistics, dropout/random masks, batch-coupled losses, data
augmentation, sample reordering, and different floating-point reduction orders
can invalidate a proposed relation. Users must either define semantics that
make those effects equivalent or treat the case as a documented non-equivalence.
Gate 5 adds no sample-coverage, distributed, framework-adapter, dashboard,
service, or general tracing behavior.
