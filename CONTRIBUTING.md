# Contributing

TrainParity 0.1 has a deliberately frozen feature set: resume equivalence,
gradient-accumulation equivalence, and finite sample-coverage policies.

## Project contracts

Contributions must:

- keep runtime behavior deterministic and non-LLM-based;
- preserve `PASS`, `FAIL`, `ABSTAIN`, and `ERROR` as distinct outcomes;
- report first observed divergence without presenting it as root cause;
- keep exact comparison and user-declared tolerance separate;
- keep the production package framework-neutral;
- include typed code, public docstrings, and tests when public behavior changes;
- avoid secrets, credentials, checkpoints, generated datasets, caches, private
  training data, and machine-local paths in commits or reports.

Do not execute untrusted integration repositories. TrainParity is not a
sandbox, and user training code runs with the caller's permissions.

## Before opening a pull request

Open an issue before proposing a behavior, API, schema, dependency, compatibility,
or feature change. Describe the observable contract, why the existing resume,
accumulation, or sample-coverage surfaces cannot express it, a clean control, a
deliberate fault, and the expected first observed boundary. Small documentation
corrections and routine community-metadata maintenance do not require an issue.

Use this workflow:

1. Open an issue when the change category above requires one.
2. Create a focused branch; do not commit directly to `main`.
3. Make the smallest change that addresses the documented contract.
4. Run validation appropriate to the risk of the change.
5. Open a pull request and describe its public-contract impact.
6. Verify hosted CI and respond to focused review feedback.
7. Wait for maintainer review and a normal merge commit.

Contributors are not expected to have access to M3. A maintainer may request
additional evidence when a change reaches a contract, device, compatibility, or
release boundary.

## Risk-based validation

Validation should match the risk introduced by the change. More evidence is not automatically better when it does not test the changed contract.

## Repository automation

- **PR and main CI** run the normal lint, type, test, build, and onboarding
  verification for the current code.
- **Weekly Validation** runs the declared CPU PyTorch compatibility smoke for
  the currently supported versions.
- **Manual deep validation** preserves historical Gate and pinned external-
  integration replay. A maintainer dispatches it only when a relevant contract
  or audit reason explicitly calls for that evidence.
- **Release workflow** is reserved for protected publication.

Ordinary documentation and community pull requests do not require manual deep
validation. An ordinary CPU bug fix does not require Gate 0-7I replay merely
because that historical audit capability exists. Historical replay is retained
for targeted audits, not as an acceptance criterion for every pull request.

GPU and M3 evidence remain driven by the changed contract. For public API,
schema, dependency, compatibility, or release changes, the maintainer may
request additional evidence proportional to the risk.

### 1. Community and repository metadata

For Issue Forms, contribution templates, conduct documents, and similar metadata:

- run `git diff --check`;
- parse or syntax-check structured files and run focused checks when applicable;
- require successful hosted CI.

M3, GPU, compatibility matrices, release checks, PyPI checks, and artifact
validation are not required by default.

### 2. README and package documentation

For installation, README examples, package metadata explanations, or public
documentation:

- run `git diff --check`;
- execute the affected README example;
- run focused release-surface and packaging tests;
- build wheel and sdist and run Twine metadata checks;
- require successful hosted CI.

A GPU run or the M3 compatibility matrix is not required unless the documented
claim itself concerns that evidence.

### 3. Runtime fixes

For a focused runtime defect:

```text
make lint
make typecheck
make test
```

Also run a focused regression and the relevant installed or source-tree
quickstart. A CPU-only change does not mechanically require a GPU run.

### 4. GPU-specific behavior

Use same-device GPU verification when the changed contract concerns CUDA RNG,
AMP or GradScaler behavior, device state, or GPU execution, or when a maintainer
requests it. Do not infer cross-device numerical equivalence.

### 5. Public API, schema, dependencies, and compatibility

Run focused contract tests, the full suite, wheel and sdist builds, Twine checks,
and relevant installed-wheel smoke tests. Provide compatibility evidence
appropriate to the changed claim and obtain maintainer review. Run the full
compatibility matrix when the supported dependency range changes, not merely
because an unrelated patch was made.

### 6. Release-only validation

Release activation and publication require the repository's release checks,
exact-wheel smoke tests outside the source tree, artifact identity verification,
and the protected Trusted Publishing workflow. Tag and GitHub Release checks
belong to the authorized release process, not ordinary pull requests.

## Licensing

By contributing, you agree that your contribution is licensed under the
repository's MIT license.
