# Gate 7 report

## Outcome

**PASS — recommendation: READY_FOR_HUMAN_REVIEW**

The frozen v0.1 release candidate is documented, packaged, audited, and verified without publication.

Gate 7 stops here for final human review. Nothing was published, tagged,
released, renamed, or made public.

## Acceptance criteria

- [x] frozen public API: stable names=30
- [x] versioned machine reports: schema=1, package=0.1.0rc1
- [x] three installed CPU examples: three clean PASS and three intentional FAIL
- [x] release documentation: README, API, validation, design, comparison, limitations, provenance, security, contribution, release notes
- [x] conservative compatibility: Python 3.11 and PyTorch 2.7 only; exact GPU fixtures documented
- [x] name availability rechecked: time-limited result recorded; no rename or publication
- [x] repository and distribution audit: no blocker; Gate evidence excluded from wheel and sdist
- [x] accepted evidence preserved: hashes unchanged=57
- [x] user remote-development document preserved: observed=6b532b94660949abec3c50bbe826d2154a797eeec744d3e5c91058dec9a96300; required working-tree hash=6b532b94660949abec3c50bbe826d2154a797eeec744d3e5c91058dec9a96300
- [x] complete verification: tests=141, coverage=90.9%
- [x] hosted CPU CI: run=31465100016 conclusion=success
- [x] no irreversible remote action: not published/tagged/released; repository remains private

## Installed-wheel examples

- `trainparity.quickstarts.resume`: clean=PASS, intentional=FAIL, first observed=`scheduler.last_epoch`
- `trainparity.quickstarts.accumulation`: clean=PASS, intentional=FAIL, first observed=`loss_accounting.denominator`
- `trainparity.quickstarts.sample_coverage`: clean=PASS, intentional=FAIL, first observed=`coverage.same_rank_duplicate`

The examples ran from outside the repository in a newly created Python
environment. Their failures are intentional first-observed divergences, not
root-cause claims.

## Distribution audit

- Wheel: `trainparity-0.1.0rc1-py3-none-any.whl`
  (49123 bytes)
- Source distribution: `trainparity-0.1.0rc1.tar.gz`
  (97977 bytes)
- Repository/distribution blockers: []
- Accepted evidence hashes checked: 57

## Exact commands

- `make lint`
- `make typecheck`
- `make test`
- `make build`
- `make release-check`
- `python scripts/verify_gate.py 0`
- `python scripts/verify_gate.py 1`
- `python scripts/verify_gate.py 2`
- `python scripts/verify_gate.py 3`
- `python scripts/verify_gate.py 4`
- `python scripts/verify_gate4_friction_audit.py`
- `python scripts/verify_gate4b.py`
- `python scripts/verify_gate5.py`
- `python scripts/verify_gate.py 6`
- `python scripts/verify_gate.py 7`
- `git diff --check`

## Remaining limitations

- The compatibility matrix is exact tested evidence, not a promise for untested Python, PyTorch, CUDA, GPU, operating-system, distributed, project, or scale combinations.
- The full-value backend favors correctness and can be expensive for large state.
- Users declare observation completeness, equivalence, sample-ID semantics, and any tolerance.
- Sample coverage proves one finite declared window and does not inspect sample contents or establish infinite-stream behavior or general shuffle quality.
- A first observed divergence is not a root-cause claim.
- TrainParity is not a sandbox; trusted code and checkpoints remain the caller's responsibility.
- The PyPI name check is time-limited and does not reserve the name; this candidate remains unpublished.
