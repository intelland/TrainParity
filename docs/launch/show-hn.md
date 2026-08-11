# Show HN draft

Title: Show HN: TrainParity – deterministic A/B checks for PyTorch training state

TrainParity checks explicit equivalence between controlled PyTorch executions.
Its resume test crosses save, process exit, fresh-process load, and continued
training; its accumulation test observes bounded phases around one optimizer
update; its coverage audit applies explicit policies to stable sample IDs.

The output is a versioned JSON PASS/FAIL/ABSTAIN/ERROR report with the first
observed divergence, not a root-cause diagnosis. Runtime is local and
deterministic, with no LLM call. The README publishes exact validation fixtures
and their limits rather than a general detection percentage.

I am looking for feedback on one question: is the small importable-case API a
reasonable price for a real fresh-process boundary, or is there a common
checkpoint structure that it cannot represent without invasive glue?

Repository: https://github.com/intelland/TrainParity
