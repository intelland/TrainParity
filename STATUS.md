# TrainParity Status

## Active gate

Gate 3 — implemented and machine-verified; stopped for human review.

## Objective

Prove equivalence between continuous training and an interrupted checkpoint /
real-process-exit / fresh-process-load / resumed execution. Establish precise
step-boundary alignment, initial-equivalence and baseline-self-consistency
prechecks, strict four-state outcomes, data-trajectory priority, and actionable
first-observed-divergence reports.

## Constraints

- Implement Gate 3 only and stop for human acceptance.
- Run Python, PyTorch, competitor, and experiment workloads on M3, not locally.
- Keep every environment, cache, checkout, log, and output under
  `/scratch/mp25/jwuu0254/zxh/TrainParity`.
- Preserve all accepted Gate 0 evidence unchanged.
- Preserve all accepted Gate 0 and Gate 1 evidence unchanged.
- Define snapshot `step=N` as state after exactly N completed optimizer updates;
  use only the `completed_training_step` phase in this Gate.
- Compare two initial snapshots before resume attribution and run an independent
  baseline self-consistency precheck; nondeterminism returns `ABSTAIN`.
- Cross a real checkpoint file boundary and a genuinely new Python process;
  record distinct pre-save and post-load PIDs and rebuild model, optimizer,
  scheduler, and scaler objects in the child process.
- Keep `PASS`, `FAIL`, `ABSTAIN`, and `ERROR` distinct. Import, launch, timeout,
  checkpoint, load, and corrupt-result failures are `ERROR`, not parity failures.
- Capture sample IDs when supplied, otherwise a deterministic batch fingerprint;
  unavailable stable data identity causes `ABSTAIN` rather than a false pass.
- Preserve every difference at the first divergent step, with deterministic
  primary ordering and no inferred root-cause claim.
- Include at least ten formal fault fixtures, including hidden module-global
  state and off-by-one resume/data-cursor cases, and repeat each stable result
  three times.
- Validate clean, CUDA RNG omission, and GradScaler omission on at least one real
  GPU inside one Slurm allocation; never compare different GPU models.
- Do not add runtime LLM/agent dependencies, distributed support, a web UI,
  service, registry, or platform functionality.
- Do not implement phase tracing, snapshot performance optimization,
  accumulation, full sample-coverage policy, or any other later-gate feature.
- Continue to describe outputs as first observed divergence, never root cause.
- Preserve the user's uncommitted `CODEX_REMOTE_DEVELOPMENT.md` changes exactly.

## Verification commands

Run from the M3 repository checkout unless noted otherwise:

```bash
make lint
make typecheck
make test
make build
python -m experiments.gate3.run_cpu_matrix \
  --output "$PROJECT_ROOT/outputs/gate3/cpu_matrix.json"
sbatch scripts/slurm_gpu_matrix.sbatch --gate 3
python scripts/verify_gate.py 3
git diff --check
```

The final Gate 3 report records initial/self-consistency evidence, process PIDs
and object identities, first divergent steps and all same-step differences,
CPU/GPU fault matrices, three-run repeatability, coverage, exact commands, and
known limitations.

## Current state

Gate 0 was accepted by the human reviewer on 2026-08-10. Its machine report is
`PASS` with recommendation `GO`; the report and all supporting evidence remain
preserved in their existing paths.

Gate 1 was accepted by the human reviewer on 2026-08-10. Its 28-line selected
adapter, wheel-installed fresh-process import, clean `PASS`, and faulty `FAIL`
evidence remain preserved.

Gate 2 was accepted by the human reviewer on 2026-08-10 based on its stable
fault paths, clean controls, immutable capture, parameter-name optimizer state,
separate comparison policies, `ABSTAIN` behavior, coverage, evidence
preservation, and hosted CI confirmation.

Gate 3 machine verification is `PASS` with recommendation `GO`. The CPU matrix
has 0 clean false positives, detects 11/11 stable CPU faults with 11/11 expected
first components, returns `ABSTAIN` for baseline nondeterminism and `ERROR` for
child failure, and records distinct pre-save/post-load PIDs. The A100 matrix in
Slurm job `58957857` passed clean, omitted CUDA RNG, and omitted GradScaler
cases three times each on one visible GPU. The complete test suite has 71
passing tests and 94.86% core coverage. Accepted Gate 0–2 evidence hashes remain
unchanged.

The first GPU attempt, job `58957648`, correctly returned `ERROR` when CUDA RNG
ByteTensors were loaded onto CUDA instead of CPU. Its raw evidence remains at
`outputs/gate3/gpu_matrix_attempt1.json`; the corrected restoration was verified
by the passing job above.

Gate 3 evidence and reports form the fully verified final checkpoint prepared
for one fast-forward from `gate-3` to `main`. No Gate 4 work has started.

No Gate 4 work is authorized.
