# Gate 0 competitor and differentiation study

## Decision summary

**Recommendation: GO to human Gate 0 review.** The experiment supports a
structural distinction rather than a naming or presentation distinction:
TrainCheck infers general runtime invariants from traces, while the proposed
TrainParity contract directly executes a user-declared A/B relation and reports
the first observed state-path divergence.

After clean-control correction, TrainCheck 0.1.2 detected the missing-scheduler
fixture but did not produce fault-specific evidence for missing RNG state,
gradient accumulation mean-of-means, or sample duplication. The 77-line
TrainParity throwaway prototype detected all four deterministically and returned
one precise first step/path for each. Three core faults therefore meet the Gate
0 requirement for output that TrainCheck did not directly provide. The
prototype is 79 physical lines (64 nonblank, noncomment lines).

This is feasibility evidence from tiny controlled cases, not a claim that
TrainParity is generally superior or that TrainCheck cannot detect variants of
these bugs with different references, instrumentation, or configuration.

## Competitor verified

- Project: [OrderLab/TrainCheck](https://github.com/OrderLab/TrainCheck)
- Package: [traincheck 0.1.2 on PyPI](https://pypi.org/project/traincheck/0.1.2/)
- Paper: [Training with Confidence, OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/jiang)
- Official usage: [TrainCheck Usage Guide](https://orderlab.io/TrainCheck/usage-guide/)
- License reported by the repository/package: Apache-2.0

The official workflow is reference trace collection, invariant inference,
target trace collection, then checking. The paper and current documentation
describe monkey-patched PyTorch API/state tracing and learned invariants. This
study exercised that workflow rather than inferring behavior from the README.

## Experiment boundary

- Host: M3 cluster, CPU login-node execution.
- Project boundary: `/scratch/mp25/jwuu0254/zxh/TrainParity`.
- Python: 3.11.15.
- PyTorch: 2.13.0+cpu from the official PyTorch CPU wheel index.
- TrainCheck: 0.1.2 from PyPI.
- TrainCheck was invoked only through its four installed CLI programs.
- The competitor repository was not executed or copied.
- All models and inputs are tiny, deterministic, and download-free.
- Raw traces/logs: `$PROJECT_ROOT/outputs/gate0/traincheck`.
- Recorded summary: `experiments/gate0/recorded/traincheck_summary.json`.

## Method

For each fault:

1. collect a short clean reference trace;
2. infer invariants with the pandas backend;
3. collect and check a second clean control using those invariants;
4. collect and check the faulty target;
5. parse each failed invariant into a stable signature consisting of
   description, relation, observed step, stage, and function;
6. subtract the clean-control signature multiset from the fault signature
   multiset;
7. count a detection only when at least one fault-specific signature remains.

The clean control is essential. Without it, all four targets appeared detected,
but three had exactly the same violations as their clean controls.

## Results

| Fault | TrainCheck clean violations | Fault violations | Fault-specific | Effective result | TrainParity first observed divergence |
|---|---:|---:|---:|---|---|
| Missing scheduler state | 14 | 27 | 13 | Detected; scheduler API-order evidence at steps 2–3 | step 2, `optimizer.lr` |
| Missing RNG state | 6 | 6 | 0 | Not detected beyond control noise | step 2, `rng.torch` |
| Accumulation mean-of-means | 5 | 5 | 0 | Not detected beyond control noise | step 0, `gradient.model.weight` |
| Sample duplication | 6 | 6 | 0 | Not detected beyond control noise | step 3, `batch.sample_ids.0` |

TrainCheck end-to-end time per case, including the clean control, was 60.635 to
65.090 seconds. All six phases per case returned zero. The initial uncorrected
totals were misleading because common violations included init-stage
`torch.library.Library.impl` relations unrelated to the injected fault.

The TrainParity prototype ran every pair three times and required the complete
result structures to match. All four outcomes and first-divergence locations
were stable. It uses exact comparison only and does not infer a root cause.

## Integration and output comparison

### TrainCheck

- install a multi-package analysis stack in the training environment;
- collect a known-good reference trace;
- infer hundreds of invariants (312 in the scheduler pilot);
- collect a target trace and run a checker;
- add a clean control to distinguish invariant noise in this study;
- interpret multiple API relation failures and not-triggered invariants;
- state proxy tracking was empty for these tiny fixtures, while API tracing was
  sufficient for inference.

### TrainParity prototype

- construct baseline and candidate from the same explicit fixture;
- capture the states relevant to the declared relation;
- exact-compare stable state paths at each boundary;
- emit the first unequal step/path plus compact values, tensor metadata,
  fingerprints, and floating-point error where applicable;
- produce no claim about root cause.

The proposed product still requires a small user adapter. Its value proposition
is control and semantic specificity, not zero-configuration observability.

## Per-fault interpretation

### Missing scheduler state

TrainCheck provided relevant scheduler call-order violations, so TrainParity
does not claim unique detection. The prototype was more concise: the first
observable state divergence was the optimizer learning rate at step 2, while
TrainCheck emitted 13 fault-specific relations in addition to 14 control
violations.

### Missing RNG state

The faulty run reset the PyTorch RNG at the resume boundary. TrainCheck's six
violations were identical to the clean control. The prototype directly compared
RNG state and reported `rng.torch` at step 2.

### Gradient accumulation mean-of-means

Unequal microbatch sizes made averaging microbatch means mathematically
different from the full-batch mean. TrainCheck's five violations matched the
clean control. The prototype reported the first changed named gradient at step
0 and its maximum absolute error.

### Sample duplication

The target changed `[0, 1, 2, 3]` to `[0, 1, 2, 2]`. TrainCheck's six
violations matched the clean control and did not report the duplicate ID. The
prototype reported `batch.sample_ids.0` at step 3 with baseline `3` and
candidate `2`.

## Noise, limitations, and threats to validity

- These are four tiny fixtures, not representative training repositories.
- TrainCheck was tested at one package/PyTorch combination and with its pandas
  trace backend. The advertised `dict` backend raised `NotImplementedError` in
  `Trace.is_stage_annotated()` and was not used for final results.
- Different TrainCheck references, multiple reference traces, relation filters,
  tensor dump modes, or future versions may improve results.
- The prototype observes only explicitly captured states and cannot find bugs
  outside that observation contract.
- TrainParity adapter cost is not validated until later Gates.
- The prototype does not yet test true process exit/resume; that is Gate 3.

## Gate 0 differentiation judgment

The distinction is structural:

```text
TrainCheck: healthy traces -> inferred invariants -> target trace -> violations
TrainParity: declared relation -> controlled A/B executions -> first state-path divergence
```

The experiment meets the Gate 0 threshold on three core faults. Proceeding to
Gate 1 is justified only after human review; no Gate 1 implementation has begun.
