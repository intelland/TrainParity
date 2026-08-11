# Technical Reddit post draft

Title: I built an explicit PyTorch resume/accumulation differential tester; looking for adversarial cases

TrainParity 0.1 is scoped to three checks: fresh-process resume equivalence,
declared full-batch versus microbatch equivalence, and finite sample-ID coverage.
It snapshots model/optimizer/scheduler/RNG or bounded accumulation phases and
returns a versioned four-state report. A failure names the first observed state
path; it deliberately does not claim root cause.

The validation suite includes 13 resume faults, eight accumulation faults,
world-size and sampler edge cases, three pinned external checkpoint paths, and
same-device A100/L40S fixtures. The limitations page calls out the tiny scale,
single-process boundary, tied-parameter ambiguity, and full-value storage cost.
There is no runtime LLM or framework-specific production adapter.

I would like technical feedback on missing adversarial controls: which realistic
resume or accumulation fault would pass the currently documented observations,
and what minimal state would expose it without turning this into general event
tracing?

Repository: https://github.com/intelland/TrainParity
