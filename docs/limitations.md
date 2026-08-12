# Limitations

- TrainParity checks only user-declared observations. Unobserved state and
  incorrect declarations can make a result incomplete.
- A first observed divergence localizes evidence; it is not an inferred root
  cause and may occur downstream of the defect.
- `PASS` applies only to the declared equivalence, policy, inputs, comparison,
  and observation window. It does not prove training correctness or model
  quality.
- Resume and accumulation execution is single-process. DDP, FSDP, DeepSpeed,
  Lightning, Transformers, elastic execution, and cross-device comparison are
  outside the 0.1 contract.
- Fresh-process resume validation needs project-specific command, checkpoint,
  and observation integration. Its cost depends on the upstream checkpoint
  interface; implicit or timestamped checkpoint locations can require a
  deterministic launcher or wrapper. Retaining that integration as a
  regression or CI test usually provides more value than a one-off diff.
- A command-oriented resume check normally runs two uninterrupted baselines
  plus the split and resumed portions of a candidate. Baseline self-consistency
  and separate process boundaries intentionally cost more than one normal run.
- The exact full-value snapshot reference favors semantic clarity over storage
  and runtime efficiency.
- Exact comparison can reject benign floating-point differences. Tolerances
  must be supplied by the user and are never inferred.
- Optimizer parameter-name mapping can be ambiguous for tied parameters; the
  honest result is `ABSTAIN` unless the user's declared observation excludes
  that ambiguous optimizer state.
- Complex batches require an explicit splitter when the default tensor-tree
  batch-dimension split would not preserve sample, mask, label, or weight
  semantics.
- Stable sample IDs must be semantically unique in the declared universe.
  TrainParity validates trajectories, not ID correctness or sample contents.
- Coverage audits apply to one declared finite observation window. Unknown
  universes cannot support exactly-once claims, and an audit does not establish
  infinite-stream behavior or general shuffle quality.
- Worker provenance is optional and is `None` / JSON `null` when unavailable.
- The reproducible validation suite uses tiny fixtures, three pinned external
  projects, Python 3.11, CPU PyTorch 2.7.0/2.10.0/2.13.0, GPU PyTorch 2.7.0,
  and two GPU models. It does not imply support for other versions, hardware,
  scale, or projects.
- TrainParity is not a sandbox. Process isolation used for equivalence testing
  is not a security boundary.
