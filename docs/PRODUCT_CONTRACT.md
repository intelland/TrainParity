# TrainParity product contract

## Definition

TrainParity is a small differential-testing library for PyTorch training
semantics. A user declares two executions that should be equivalent;
TrainParity runs them from the same logical initial condition under a controlled
plan and reports whether their observed trajectories satisfy the declared
comparison policy.

TrainParity does not infer correctness from a healthy run and does not claim to
detect arbitrary training bugs.

## Contract inputs

A check must receive:

1. an importable case adapter, not an arbitrary script;
2. an explicit equivalence relation such as resume equivalence, full-batch
   versus accumulation equivalence, or a sample-coverage policy;
3. deterministic construction inputs, including a seed;
4. the execution boundaries to observe;
5. an explicit comparison policy: exact, or user-supplied `rtol`, `atol`, and
   `equal_nan` behavior;
6. every capability required by that check, such as save/load or stable sample
   IDs.

Missing or ambiguous required inputs produce `ABSTAIN` or `ERROR`, never a
fabricated `PASS`.

## Execution contract

- Baseline and candidate begin from the same logical initial condition.
- A baseline self-consistency precheck must succeed before candidate comparison.
- Resume equivalence must terminate and reload in a fresh Python process once
  Gate 3 authorizes that implementation.
- Temporary files and child processes are isolated and cleaned safely.
- The first release scope is single-process and single-device.
- Device comparisons are within one device/runtime configuration, not across
  GPU models.
- TrainParity invokes no LLM, network service, database, registry, or agent at
  user runtime.

## Observation contract

The eventual core may observe only explicitly supported state:

- batch fingerprint or stable sample IDs;
- Python, NumPy (when installed), PyTorch CPU, and PyTorch CUDA RNG state;
- named model parameters, buffers, and gradients;
- optimizer state canonicalized by stable parameter name;
- scheduler and GradScaler state;
- learning rate, loss, and user-registered metrics;
- user-registered `state_dict` objects.

An optimizer state that cannot be mapped unambiguously to parameter names causes
`ABSTAIN`. Raw parameter IDs are not a stable public state path.

## Comparison contract

- Exact comparison and explicit tolerance comparison are separate policies.
- TrainParity never guesses or silently widens a tolerance.
- Shape, dtype, device metadata, NaN, Inf, signed zero, missing keys, and extra
  keys have explicit behavior.
- Recursive traversal and state paths are deterministic.
- A mismatch report names the earliest observed step and stable path, provides
  compact baseline/candidate summaries, and includes numerical error when
  meaningful.
- The report says **first observed divergence**, not root cause.

## Result contract

Exactly one of four outcomes is returned:

- `PASS`: baseline is self-consistent, required observations succeeded, and the
  candidate satisfies the declared policy.
- `FAIL`: baseline is self-consistent and the candidate first diverges at a
  reproducible observed step/path.
- `ABSTAIN`: the relation cannot be judged reliably, for example because the
  baseline is nondeterministic or a required state is unavailable.
- `ERROR`: the test infrastructure or adapter failed, for example a child
  process crash or invalid checkpoint.

These outcomes are never collapsed into a Boolean in machine reports. A pytest
assertion may translate them into user-facing test behavior while preserving
the original result.

## Primary user experience

The primary interface is pytest plus a small typed adapter. A thin CLI may call
the same API. TrainParity does not provide a trainer and does not require
instrumentation to remain in production training.

For a conventional single-device PyTorch loop, Gate 1 must demonstrate an
importable adapter with at most 30 logical lines. If real integrations routinely
exceed that cost, the project must be reconsidered.

## Supported scope by Gate

Gate 0 proves only the product distinction with throwaway code. No production
API is implemented yet.

- Gate 1 may define the package skeleton and select an adapter API.
- Gate 2 may implement snapshot/canonicalization/comparison.
- Gate 3 may implement single-process/single-GPU resume equivalence.
- Accumulation and sample coverage remain unauthorized until Gates 5 and 6.

Passing Gate 0 does not imply that later capabilities already exist.

## Non-goals and prohibited claims

TrainParity does not:

- accept or understand arbitrary Python training scripts;
- find every training bug or determine whether a model trains well;
- infer invariants from reference traces;
- provide an AI diagnosis or claim root cause;
- provide monitoring, dashboards, services, registries, leaderboards, or SaaS;
- support DDP, FSDP, DeepSpeed, Lightning, or Transformers in the initial scope;
- compare different GPU models bitwise;
- auto-tune tolerance, performance, memory, or hyperparameters.

Prohibited descriptions include “universal training doctor,” “one-click
debugging,” and “works on any Python training script.”

## Failure and stop conditions

The project must stop or be reworked if later evidence shows any of the
following:

- TrainCheck or another existing tool supplies the same explicit A/B and
  first-divergence workflow at equal or lower integration cost;
- clean fixtures produce unexplained false positives;
- stable first step/path reporting cannot be maintained;
- adapter median logical LOC exceeds 30 across the Gate 4 integrations;
- the product requires a service/platform or broad framework adapters to be
  useful;
- most realistic baselines are too nondeterministic to compare.

## Gate 0 evidence boundary

The Gate 0 prototype is intentionally throwaway and exact-only. It demonstrates
the desired report shape on four deterministic fixtures but is not production
code, a public API, a process-isolated resume runner, or evidence of arbitrary
PyTorch compatibility.
