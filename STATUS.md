# TrainParity Status

## Active gate

Gate 4B production-integration-surface verification is complete with outcome
`GO`; awaiting final human review. Gate 5 has not been started.

## Objective

Move only generic resume-test orchestration into the production library, retain
small explicit project-semantics adapters, and reproduce all three pinned Gate 4
projects from fresh clones within the Gate 4B LOC, wall-time, artifact, behavior,
and evidence-preservation thresholds.

## Constraints

- Implement Gate 4B only and stop for human acceptance.
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

## Verification commands

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
python scripts/verify_gate4_friction_audit.py
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

Gate 4 completed on the dedicated `gate-4` branch. The formal M3 L40S matrix in
Slurm job `58960426` passed all three clean controls and detected all three
faults. All adapters are 24 logical lines (median 24); project-specific glue is
48, 57, and 44 logical lines; upstream modified LOC is zero. The exact external
commits, license hashes, original checkpoint paths, resource measurements, and
LOC explanations are recorded in `docs/GATE4_INTEGRATIONS.md` and
`artifacts/gate_reports/gate_4.*`.

The first observed fault paths are `scheduler.last_epoch` for the ImageNet
recipe, `best_val_loss` for nanoGPT, and `lr_scheduler.last_epoch` for Ignite.
These are observations, not root-cause claims. The machine verifier returns
`PASS` with recommendation `GO`. Gate 5 has not been started and remains
unauthorized pending human review.

The GitHub Actions workflow cloned the pinned nanoGPT commit and ran a real Gate
4 clean/fault case before `verify_gate.py 4`. Pull-request run `31414864786` for
commit `59876afdd744ca40e2add625113320fc75168385` completed successfully, including
lint, strict typing, 76 tests, build, Gate 3 verification, the pinned nanoGPT
integration, and Gate 4 verification. The hosted result is recorded in
`experiments/gate4/recorded/ci.json`.

The Gate 4 friction rework completed on M3 L40S Slurm job `58962334`. All three
fresh exact-commit clones imported TrainParity from an isolated wheel target,
used only three added user source files, retained zero tracked upstream changes,
and passed baseline self-consistency plus clean save/exit/new-process/load/resume.
The first audit attempt, job `58962047`, is preserved as `ERROR`: its stronger
Ignite control did not inherit `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; this was an
infrastructure compatibility error, not a parity result.

The honest user-required LOC totals are 266 for pytorch/examples ImageNet, 275
for nanoGPT, and 271 for Ignite. Adapter LOC remains small (13, 13, and 15), but
the shared user-visible command/process/snapshot orchestration makes supporting
glue 253, 262, and 256 LOC. End-to-end multipliers are 15.91x, 4.03x, and 3.85x;
the prior comparator-only timing remains recorded but is no longer presented as
total overhead. The 116-line closer hand-written Ignite check covers model,
optimizer, scheduler, and torch CPU RNG and reports the first observed fault
divergence at `$.scheduler.last_epoch`.

The dedicated audit report recommends `REWORK`: technical GO evidence remains
positive, but current external command-oriented integration friction is not yet
acceptable without a human decision. No production API, framework adapter,
backend optimization, new project, or Gate 5 code was added. Final verification
passed Ruff, Mypy, 79 tests at 94.86% coverage, build, the accepted Gate 4
verifier, the dedicated friction verifier, and `git diff --check`.

Gate 4B moved only framework-neutral process orchestration into production.
Fresh-clone user integration is now 30, 31, and 32 logical LOC (median 31),
with adapter maxima of 19 LOC, supporting glue of 13 LOC, and zero upstream
modifications. M3 L40S Slurm job `58970083` passed 3/3 clean controls and
detected 3/3 faults; total wall multipliers were 5.261x, 5.202x, and 4.949x,
and all artifact thresholds passed. The profiled ImageNet full-value snapshot
path improved from 22.278513s to 0.183875s without changing exact bytes or
comparison semantics. Full tests passed 97/97 at 94.27% coverage.

Hosted GitHub Actions run `31437964382` at commit
`5b8a84de2df502252d73e6eb806f6371f29e42ce` passed lint, type-check, tests,
build, isolated Gate 0-4 and friction verifier replays, the pinned Gate 4 case,
and a wheel-installed Gate 4B nanoGPT fresh-clone case. The final dedicated
Gate 4B verifier returns `PASS`, the report recommends `GO`, accepted evidence
remains preserved, and Gate 5 remains unauthorized.

Gate 4B was accepted by the human reviewer on 2026-08-11. Gate 5 is now the
only authorized scope and is under development on branch `gate-5`. The formal
equivalence relation and one-update boundary are frozen in
`docs/GATE5_ACCUMULATION_CONTRACT.md`. Gate 6 and all distributed,
sample-coverage, framework-adapter, dashboard, and service work remain
unauthorized.

Gate 5 implementation and M3 evidence are complete. The
formal CPU matrix passed 30/30 expected outcomes across three repeats; same-
device L40S Slurm job `58980407` passed 12/12, including clean AMP and the AMP
unscale/GradScaler timing fault. All eight required faults were detected, all
formal executions began from verified-equal state in three distinct fresh
processes, and clean false positives were zero. Full regression is 107 tests at
92.54% coverage. Fresh pinned ImageNet and nanoGPT product checks pass at 38
and 37 total user logical LOC with zero upstream modifications. Gate 6 has not
been started.

Hosted GitHub Actions run `31453089659` for PR `#2` at commit
`ee7951f67052ad965b4c26188c048e2401742077` passed all steps, including 107
tests, build, isolated Gate 0-4 replay, Gate 4/4B external checks from their
accepted package versions, the dev6 wheel-installed nanoGPT Gate 5 surface,
and the dedicated Gate 5 verifier.

Gate 5 was accepted by the human reviewer on 2026-08-11. Gate 6 is now the
only authorized scope and is an optional sample-coverage inclusion decision.
The implementation must audit only user-supplied stable sample IDs under the
explicit `exactly_once`, `at_least_once`, `no_cross_rank_overlap`, and
`expected_padding` policies. Unknown expected universes must `ABSTAIN`; same-
rank duplication and cross-rank overlap remain distinct; terminal summaries
must be bounded while optional machine evidence remains complete. Gate 6 must
not add a distributed launcher, training integration, checkpointing, GPU work,
framework adapter, dashboard, service, release work, or later expansion.

The Gate 5 nanoGPT product fixture excludes the shared
`transformer.wte.weight` / `lm_head.weight` tensor from the optimizer, so its
optimizer parameter groups and optimizer state contain neither alias. All
other unique model parameters in the fixture are included. Full model and
gradient phase observations use `named_parameters(remove_duplicate=False)` and
therefore still include both tied aliases, including the tied gradient. No
project-specific optimizer mapping rule was added. The Gate 5 verifier already
confirmed that every accepted Gate 0-4B evidence hash remained unchanged; this
record closes that carry-forward without rerunning GPU work.

Gate 6 verification commands are `make lint`, `make typecheck`, `make test`,
`make build`, `python scripts/verify_gate.py 0` through `4`,
`python scripts/verify_gate4_friction_audit.py`,
`python scripts/verify_gate4b.py`, `python scripts/verify_gate5.py`,
`python scripts/verify_gate.py 6`, the dedicated Gate 6 verifier, and
`git diff --check`. The final report must conclude `INCLUDE_MODULE`,
`REWORK_MODULE`, or `OMIT_MODULE` and then stop for human review.
