# TrainParity Status

## Active gate

Gate 4 — real-project integration and product-friction Go/No-Go.

## Objective

Measure TrainParity integration effort and diagnostic value against three real,
commit-pinned external PyTorch recipes: image classification, language modeling,
and an Engine-based loop with extra resumable state. Exercise each upstream
project's own checkpoint save/load path with clean and realistic faulty resume
runs, without optimizing the snapshot backend.

## Constraints

- Implement Gate 4 only and stop for human acceptance.
- Run Python, PyTorch, competitor, and experiment workloads on M3, not locally.
- Keep every environment, cache, checkout, log, and output under
  `/scratch/mp25/jwuu0254/zxh/TrainParity`.
- Preserve all accepted Gate 0 evidence unchanged.
- Preserve all accepted Gate 0 and Gate 1 evidence unchanged.
- Pin three external repositories to exact commits and record their licenses;
  keep at least two projects external rather than copying their code.
- Exercise original upstream checkpoint save/load implementations; do not replace
  them with TrainParity-authored clean checkpoint routines.
- Record adapter, supporting glue, upstream modified, and total integration LOC.
- Target median adapter logical LOC <= 30 and explain integrations over 50 total LOC.
- Use tiny generated data and one realistic resume fault per project; adapters
  must not specialize their comparison behavior to the injected fault.
- Compare integration effort and diagnostics with a minimal hand-written test.
- Measure runtime, peak memory, checkpoint/snapshot size, and comparison overhead.
- Do not add runtime LLM/agent dependencies, distributed support, a web UI,
  service, registry, or platform functionality.
- Do not implement framework-specific production adapters, distributed support,
  later checks, a dashboard, service, or snapshot optimization.
- Continue to describe outputs as first observed divergence, never root cause.
- Preserve the user's uncommitted `CODEX_REMOTE_DEVELOPMENT.md` changes exactly.

## Planned verification commands

Run from the M3 repository checkout unless noted otherwise:

```bash
make lint
make typecheck
make test
make build
python -m experiments.gate4.run_matrix \
  --external-root "$PROJECT_ROOT/external/gate4" \
  --output "$PROJECT_ROOT/outputs/gate4/matrix.json"
python scripts/verify_gate.py 4
git diff --check
```

The final Gate 4 report will record repository commits and licenses, original
checkpoint paths, clean/fault results, LOC categories, upstream diffs, hand-test
comparison, runtime/memory/artifact measurements, CI evidence, and a candid
GO/REWORK/STOP recommendation.

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

Gate 3 was accepted by the human reviewer on 2026-08-11. Its CPU matrix
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

Gate 4 is authorized and in progress on the dedicated `gate-4` branch. The
candidate external repositories are `pytorch/examples`, `karpathy/nanoGPT`, and
`pytorch/ignite`; commit and license evidence is being verified before adapters
are implemented. No Gate 5 work is authorized.
