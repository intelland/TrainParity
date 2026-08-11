# Contributing

TrainParity 0.1 has a deliberately frozen feature set: resume equivalence,
gradient-accumulation equivalence, and finite sample-coverage policies.

Before proposing a change, open an issue describing the observable contract,
why existing public objects cannot express it, clean controls, deliberate fault
fixtures, and the expected first observed divergence. Feature work outside the
frozen surface should not be mixed with fixes or documentation changes.

Contributions must:

- keep runtime behavior deterministic and non-LLM-based;
- preserve `PASS`, `FAIL`, `ABSTAIN`, and `ERROR` as distinct outcomes;
- avoid root-cause claims;
- keep exact and explicit-tolerance comparison separate;
- include typed code, public docstrings, and tests;
- avoid framework-specific branches in the production package;
- avoid secrets, checkpoints, generated datasets, caches, and machine-local
  paths in commits.

Run the repository checks on a supported Python 3.11 environment:

```text
make lint
make typecheck
make test
make build
make release-check
```

Do not execute untrusted integration repositories. By contributing, you agree
that your contribution is licensed under the repository's MIT license.
