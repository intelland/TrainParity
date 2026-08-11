# Comparison with TrainCheck

[TrainCheck](https://github.com/OrderLab/TrainCheck) is an OrderLab project that
infers training invariants from a reference execution and checks them against a
target trace. Its current package and usage should be evaluated from its
[official repository](https://github.com/OrderLab/TrainCheck) and
[PyPI page](https://pypi.org/project/traincheck/).

TrainParity takes a different structural approach: the user declares an A/B
equivalence relation, the relevant observations, and any numeric tolerance.
Resume tests deliberately cross checkpoint, exit, and fresh-process load
boundaries. Accumulation tests compare bounded phases around one optimizer
update. Coverage tests apply explicit policies to finite sample-ID
trajectories.

The distinction matters diagnostically. An inferred invariant can detect that
a run violates behavior learned from its reference. A differential test can
show the earliest observation where two controlled executions cease to match.
Neither result by itself establishes root cause, and neither tool is a
universal detector.

Gate 0 compared the approaches using controlled clean and four-fault prototype
evidence. That evidence supported building the explicit differential design; it
was not a benchmark proving general superiority, and no TrainCheck source was
copied into TrainParity.

TrainParity's additional costs are real: users must express project semantics,
the full-value reference stores explicit state, and a differential check runs
multiple executions. TrainCheck and TrainParity should therefore be selected
for the contract a user actually needs, not from a headline detection count.
