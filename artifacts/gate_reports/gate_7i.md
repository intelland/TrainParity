# Gate 7I release-surface hardening report

## Outcome

**PASS — READY_FOR_FINAL_PUBLICATION_REVIEW**

Publication remains held. No PyPI upload, Git tag, GitHub release, rename, merge,
or repository-visibility change was performed.

## Public surface

- Recommended top-level names: 11
- Advanced `trainparity.api` names: 26
- Removed accidental public names: ExternalProcessEvidence, PACKAGE_VERSION, SampleCoverageAuditor
- README integration: 30 logical LOC,
  complete PyTorch DataLoader case, run directly by CI
- Console-script entry point: absent for v0.1

## Compatibility

- Python 3.11.15 / PyTorch 2.7.0+cpu: PASS (3 clean PASS, 3 intentional FAIL)
- Python 3.11.15 / PyTorch 2.10.0+cpu: PASS (3 clean PASS, 3 intentional FAIL)
- Python 3.11.15 / PyTorch 2.13.0+cpu: PASS (3 clean PASS, 3 intentional FAIL)

The declared range is `torch>=2.7,<2.14`. No claim is
made for PyTorch 2.14, another Python release, or an unrecorded device.

## Verification

- Tests: 146 passed
- Coverage: 90.30837004405286%
- Installed-wheel smoke: PASS
- Repository/archive audit: PASS
- Twine metadata/README rendering: PASS
- Accepted evidence hashes unchanged: True
- User document SHA-256 unchanged: 6b532b94660949abec3c50bbe826d2154a797eeec744d3e5c91058dec9a96300

## Workflow boundary

- `ci.yml`: read-only PR checks, no secret or OIDC access
- `validation.yml`: manual/scheduled full replay and compatibility matrix
- `release.yml`: manual default-branch job, protected `pypi` environment,
  job-scoped OIDC, no PR artifact consumption; never executed
- Every third-party action is pinned to a verified full commit SHA and checkout
  credentials are not persisted.

## Validation-language statement

Detection counts are presented only as results from the pinned reproducible
validation suite. They are not presented as universal detection rates.

## Exact commands

- `make lint`
- `make typecheck`
- `make test`
- `python -m pytest -q --no-cov examples/test_readme_case.py`
- `python scripts/verify_gate.py 0`
- `python scripts/verify_gate.py 1`
- `python scripts/verify_gate.py 2`
- `python scripts/verify_gate.py 3`
- `python scripts/verify_gate.py 4`
- `python scripts/verify_gate4_friction_audit.py`
- `python scripts/verify_gate4b.py`
- `python scripts/verify_gate5.py`
- `python scripts/verify_gate6.py`
- `python scripts/verify_gate7.py --allow-pending-ci`
- `sbatch scripts/slurm_gate7i_compatibility.sbatch`
- `make release-check`
- `python -m scripts.build_gate7i_bundle`
- `python scripts/verify_gate7i.py`
- `git diff --check`

## Remaining limitations

- A first observed divergence is evidence, not a root-cause claim.
- Resume and accumulation remain single-process checks; sample coverage observes
  declared finite windows and does not launch distributed training.
- Full-value snapshots prioritize exactness over performance.
- The release workflow still requires a human to configure protected-environment
  reviewers and explicitly approve publication.
