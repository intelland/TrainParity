# TrainParity Status

## Active gate

Gate 7 release-candidate preparation is complete with status `PASS` and
recommendation `READY_FOR_HUMAN_REVIEW`. No release, tag, publication,
repository visibility change, or later Gate is authorized.

## Objective

Freeze and document the v0.1 public surface for resume equivalence,
gradient-accumulation equivalence, and sample-coverage policies; prepare and
audit an honest installable release candidate without publishing it.

## Constraints

- Implement Gate 7 only and stop for final human acceptance.
- Run Python, PyTorch, competitor, and experiment workloads on M3, not locally.
- Keep every environment, cache, checkout, log, and output under
  `/scratch/mp25/jwuu0254/zxh/TrainParity`.
- Preserve all accepted Gate 0-6 evidence unchanged.
- Freeze the feature set; add no check, framework adapter, service, dashboard,
  registry, distributed launcher, or general event tracer.
- Keep stable public objects distinct from internal runners, backends,
  experiments, and Gate tooling.
- Put machine-schema and package versions in JSON reports.
- Ship three tiny CPU examples, each with a clean `PASS` and intentional `FAIL`.
- Test the built wheel from outside the repository in a clean environment.
- Audit source and distributions for secrets, local metadata, large data, and
  unintended runtime contents without deleting accepted evidence.
- Report name availability but do not infer reservation or rename the project.
- Do not publish to PyPI, create a tag/release, or change repository visibility.
- Do not add runtime LLM/agent dependencies, distributed support, a web UI,
  service, registry, or platform functionality.
- Continue to describe outputs as first observed divergence, never root cause.
- Preserve the user's uncommitted `CODEX_REMOTE_DEVELOPMENT.md` changes exactly.

## Verification commands

Run from the M3 repository checkout unless noted otherwise:

```bash
make lint
make typecheck
make test
make build
make release-check
python scripts/verify_gate.py 0
python scripts/verify_gate.py 1
python scripts/verify_gate.py 2
python scripts/verify_gate.py 3
python scripts/verify_gate.py 4
python scripts/verify_gate4b.py
python scripts/verify_gate5.py
python scripts/verify_gate.py 6
python scripts/verify_gate.py 7
git diff --check
```

The final Gate 7 report records the frozen API, example outcomes, exact
compatibility evidence, source/distribution audit, clean-environment wheel
test, package/repository-name check, historical verifier results, hosted CI,
and remaining limitations. No criterion is blocked.

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

Gate 6 implementation and evidence are complete. The CPU-only matrix matched
17/17 expected four-state outcomes and covers world sizes 1/2/3/4,
non-divisible lengths, `drop_last`, declared padding, missing IDs, same-rank
duplication, cross-rank overlap, a custom sampler, finite `IterableDataset`,
unknown-universe `ABSTAIN`, and multi-epoch shuffle windows. Full regression is
134 tests at 92.46% coverage. SequentialSampler and DistributedSampler product
surfaces pass at 15 and 18 user logical LOC with zero upstream modifications.

The 11-logical-line Counter baseline reports flat missing and duplicate IDs.
TrainParity adds seven recorded structural benefits: rank/worker/epoch/position
provenance, expected-padding interpretation, finite-universe missing detection,
same-rank versus cross-rank separation, resume-cursor coverage, deterministic
first observed violations with four-state JSON, and bounded summaries with
optional complete evidence. The Gate 6 recommendation is `INCLUDE_MODULE`.

Draft PR `#3` targets `gate-5`. Hosted GitHub Actions run `31456866444` for
commit `b219358dbccfc17c235ef272c66bfadd2d28b7ed` completed successfully,
including all prior verifiers and the Gate 6 matrix, product surface, and
dedicated verifier. Gate 7 has not been started and remains unauthorized.

Gate 6 was accepted with `INCLUDE_MODULE` by the human reviewer on 2026-08-11.
Gate 7 is now the only authorized scope and is release-candidate preparation
for the frozen v0.1 feature set: resume equivalence, gradient-accumulation
equivalence, and sample-coverage policies. No new check, framework adapter, or
later expansion is authorized.

The Gate 7 documentation must define a stable sample ID as semantically unique
within the user-declared expected universe. TrainParity validates ID
trajectories, never sample contents. Missing worker provenance is represented
as unavailable (`None`/JSON `null`), never worker 0. One audit establishes only
the declared finite observation window and does not establish sample contents,
infinite-stream exactly-once behavior, or general shuffle quality.

Gate 7 must freeze a documented stable public API, add schema/package versions
to production JSON, ship three minimal CPU PASS/FAIL examples, prepare an
evidence-first README and release/security/contribution documentation, recheck
name availability, validate a built wheel outside the repository, publish only
tested compatibility claims, audit source/distributions for sensitive or
unintended content, and preserve accepted evidence plus the user's uncommitted
`CODEX_REMOTE_DEVELOPMENT.md`. Verification is `make release-check`, every
accepted Gate verifier, `python scripts/verify_gate.py 7`, and hosted CPU CI.
PyPI publication, tags/releases, visibility changes, automatic renaming, and
all irreversible remote release actions remain prohibited.

Gate 7 implementation and verification are complete. The frozen public surface
contains 30 top-level names including `__version__`; public reports use schema
version 1 and package version `0.1.0rc1`. All three installed CPU examples pass
their clean case and fail their intentional case at the expected first observed
path. M3 Slurm job `58995900` completed `make release-check`: Ruff, strict Mypy,
141 tests at 90.90% coverage, wheel/sdist build, new-environment wheel install,
three examples outside the repository, and the zero-blocker release audit.

The final archives are `trainparity-0.1.0rc1-py3-none-any.whl` and
`trainparity-0.1.0rc1.tar.gz`. Gate experiments, accepted evidence, development
plans, launch drafts, and historical Gate documents are excluded. The audit
found no secret-like credentials or release-facing local metadata. Historical
machine/user paths remain only in preserved evidence, development provenance,
and repository-only tooling.

Draft PR `#4` targets `gate-6`. Hosted GitHub Actions run `31465100016` at
commit `8fba5a5cce2090272ef971e54f2a04237cc3953f` passed every configured step,
including isolated Gate 0-4 replay, Gate 4B/5/6 verification, external product
surfaces, `make release-check`, the three installed examples, report generation,
and Gate 7 verification. The final Gate 7 verifier reports `PASS`, 57 unchanged
accepted-evidence hashes, and the exact user document hash
`6b532b94660949abec3c50bbe826d2154a797eeec744d3e5c91058dec9a96300`.

The PyPI endpoint returned 404 for `trainparity` at the recorded check time,
and GitHub name search found only the existing private owner repository. These
checks do not reserve the name. Nothing was published, tagged, released,
renamed, merged, or made public. Gate 7 stops for final human review.

Gate 7 was accepted as a release candidate on 2026-08-11, with publication
held. Gate 7I is the only authorized scope: harden the public import surface,
replace the README placeholder-style introduction with a complete CI-tested
case, turn `docs/api.md` into a signature-level reference, verify a current
PyTorch compatibility range, split and least-privilege-harden GitHub Actions,
prepare (but never execute) a protected-environment Trusted Publishing
workflow, correct maintainer/release metadata, document coverage exclusions,
and create a real-path human-review bundle.

Gate 7I must not add checks or adapters, alter comparison semantics, publish to
PyPI, create a tag or GitHub release, rename the project, change repository
visibility, or consume untrusted PR artifacts in a release job. The current
branch is `gate-7i`. The user's uncommitted `CODEX_REMOTE_DEVELOPMENT.md` must
remain byte-identical at SHA-256
`6b532b94660949abec3c50bbe826d2154a797eeec744d3e5c91058dec9a96300`.

Planned verification is Ruff, strict Mypy, unit/contract/integration tests,
every accepted Gate verifier, Python 3.11 with PyTorch 2.7.0/2.10.0/2.13.0,
wheel and sdist build, installed-wheel smoke outside the repository, all three
quickstarts, Twine metadata/README rendering checks, workflow contract tests,
repository/archive security audit, link inventory verification, and hosted
CI/validation where their non-release triggers permit. Gate 7I stops for human
publication review.
