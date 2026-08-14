# Design

TrainParity evaluates explicit observations rather than trying to interpret an
arbitrary training program. Its frozen 0.1 surface contains three checks.

## Resume

The user supplies an importable command-oriented case with the project's real
checkpoint location and observation semantics. TrainParity plans a continuous
execution and an interrupted execution, stages the checkpoint, exits the first
process, loads in a fresh process, captures immutable snapshots, and compares
aligned completed-step observations. Baseline self-consistency is checked
first. Ambiguous or unavailable state produces `ABSTAIN`; execution failure is
`ERROR`. The same user-declared exact or explicit-tolerance policy governs both
baseline self-consistency and the resumed-candidate comparison.

## Accumulation

The user declares whether a full batch and an explicitly split microbatch plan
should be equivalent. A logical step is one optimizer-update boundary.
TrainParity observes only loss accounting, gradients after backward and before
the optimizer step, optimizer and parameter state after the step, and scheduler
state. Numerator/denominator loss accounting is optional but explicit; its
absence is reported rather than inferred.

## Sample coverage

The user supplies ordered stable-ID observations and one of four policies.
TrainParity records rank, optional worker, epoch, and position provenance,
separates same-rank duplicates from cross-rank overlap, and interprets declared
padding. Unknown finite universes cause the policies that require them to
`ABSTAIN`.

## Comparison and reports

Stable paths and deterministic traversal select the first observed difference.
All differences at that boundary remain machine-readable. Exact comparison and
explicit tolerance comparison are separate policies; a tolerance is never
learned from the data. Exact floating/complex tensor differences retain exact
outcomes while also reporting absolute and relative error magnitude. Reports
have four outcomes and carry both schema and package versions; report schema 2
records resume comparison policy and tolerance values.

`FullValueBackend` is the correctness reference. It freezes tensors rather than
retaining mutable aliases and represents missing, `None`, empty, zero, NaN, and
Inf distinctly. Optimizer state is canonicalized by parameter name and
ambiguous mappings abstain.

The runtime package does not contain project-specific adapters, Gate
experiments, accepted evidence, launch tooling, an LLM, or an external service.
