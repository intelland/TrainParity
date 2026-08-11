# PyTorch Forums draft

Title: Feedback wanted: explicit differential checks for PyTorch resume and accumulation

I am preparing TrainParity 0.1, a small library that compares user-declared
training semantics across fresh-process resume, accumulation plans, and finite
sample-ID coverage. It reports the first observed divergence and keeps PASS,
FAIL, ABSTAIN, and ERROR distinct; it does not infer root causes or invoke an
LLM at runtime.

The pinned validation suite detects missing scheduler/RNG/optimizer state,
microbatch step-timing faults, mean-of-means loss accounting, and sample-policy
violations. Three external resume cases use their original checkpoint paths
with 30-32 lines of user integration after generic orchestration moved into the
library.

I would especially value feedback from people maintaining custom PyTorch
trainers: which state or phase in the documented adapter contract is hardest to
expose without changing your training code? Concrete counterexamples to the
single-process assumptions would help shape the post-0.1 boundary.

Repository: https://github.com/intelland/TrainParity
