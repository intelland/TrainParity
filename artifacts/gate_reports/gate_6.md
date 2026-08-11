# Gate 6 report

## Outcome

**PASS — module decision: INCLUDE_MODULE**

Explicit coverage policies add structured distributed provenance and padding semantics beyond the Counter baseline without owning distributed execution.

## Acceptance criteria

- [x] 17/17 CPU matrix rows matched their expected four-state outcome
- [x] world sizes 1/2/3/4 and every required sampler/anomaly condition were exercised
- [x] unknown-universe exactly_once returned ABSTAIN without consuming the stream
- [x] expected padding reported repeated IDs, ranks, and exact actual/declaration counts
- [x] two PyTorch sampler surfaces passed at 15 and 18 user logical LOC with zero upstream changes
- [x] seven structural benefits exceed the flat 11-line Counter baseline
- [x] accepted Gate 0-5 evidence and the user's uncommitted remote-development document remain preserved

## Product surface and Counter baseline

SequentialSampler requires 15 logical LOC and DistributedSampler 18; both modify zero upstream lines. The 11-line Counter baseline reports only flat missing/duplicate IDs. TrainParity additionally preserves rank/worker/epoch/position provenance, interprets expected padding, distinguishes same-rank duplication from cross-rank overlap, detects finite-universe missing IDs and resume-cursor anomalies, identifies a deterministic first observed violation, and returns bounded four-state machine evidence.

## Scope

The module consumes stable IDs; it does not launch DDP, Slurm, NCCL, ranks, workers, or training. Unknown-universe exactly-once claims ABSTAIN. Complete anomaly trajectories are written separately when requested, while terminal summaries remain bounded. First observed violations are not root-cause claims. CPU execution was sufficient; no GPU work was added.

## Gate 5 carry-forward

The nanoGPT tied parameter remains excluded only from optimizer groups/state; both model aliases and both gradient aliases remain observed. Accepted Gate 0-4B hashes were already verified unchanged, and no GPU work was rerun for that reporting clarification.

## Exact commands

- `make lint`
- `make typecheck`
- `make test`
- `make build`
- `python -m experiments.gate6.run_matrix --output outputs/gate6/cpu_matrix.json`
- `python -m experiments.gate6.run_product_surface --output outputs/gate6/product_surface.json`
- `python scripts/verify_gate.py 0`
- `python scripts/verify_gate.py 1`
- `python scripts/verify_gate.py 2`
- `python scripts/verify_gate.py 3`
- `python scripts/verify_gate.py 4`
- `python scripts/verify_gate4_friction_audit.py`
- `python scripts/verify_gate4b.py`
- `python scripts/verify_gate5.py`
- `python scripts/verify_gate.py 6`
- `git diff --check`

## Remaining limitations

- The auditor trusts the user-declared expected universe and stable sample-ID extractor.
- Worker provenance is optional because ordinary parent-process DataLoader iteration does not expose it.
- Each call audits one finite declared window; it does not prove sample contents or general shuffle equivalence.
- The production module is materially larger than the Counter baseline because it retains four policies, provenance, deterministic evidence, and four-state errors.
- No DDP launcher, distributed trainer, checkpoint system, GPU path, dashboard, service, or release work was added.
