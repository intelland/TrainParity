# AGENTS.md — TrainParity

## Mission

Maintain TrainParity as a small, deterministic PyTorch differential-testing
library for user-declared training equivalence and first observed divergence.
TrainParity 0.1.0 is in stable maintenance.

## Start here

Before changing the project, read:

1. `README.md`
2. `CONTRIBUTING.md`
3. `docs/api.md`
4. `docs/design.md`
5. `docs/limitations.md`

These files define the current product and maintenance contract.

## Product invariants

- Keep `PASS`, `FAIL`, `ABSTAIN`, and `ERROR` distinct.
- Report the first *observed* divergence; do not infer a root cause.
- Keep exact comparison separate from explicit user-specified tolerance.
- Never weaken predicates, tolerances, or fault fixtures to make tests pass.
- Keep the production package framework-neutral and dependencies minimal.
- TrainParity runtime must not call an LLM, agent, web service, or registry.
- Do not claim support for arbitrary Python training scripts.
- Do not add general Lightning, Transformers, DeepSpeed, DDP, or FSDP adapters
  without an explicitly authorized public-contract proposal.
- Do not copy code from OrderLab/TrainCheck.

## Maintenance workflow

- Do not proactively start 0.1.1 or a new feature without a concrete issue or
  maintainer instruction.
- Use a focused branch and make the smallest change that addresses the evidence.
- Add or update typed tests for public behavior.
- Follow the risk-based validation tiers in `CONTRIBUTING.md`.
- Run specialized device, compatibility, or release evidence only when the
  changed contract requires it.
- Prefer repository commands exposed through the `Makefile`.
- Inspect staged and unstaged changes and commit only authorized paths.
- Submit a focused PR, wait for required verification, obtain maintainer
  review, and use a normal merge commit when merge is authorized.
- Do not publish packages, create tags or releases, or change repository
  visibility without explicit authorization.

## Repository safety

- Do not add internal development journals, machine-specific instructions, or
  maintainer infrastructure metadata to the public tree.
- Historical provenance belongs in Git history and release tags, not current
  maintenance files.
- Do not commit credentials, private data, maintainer-local paths, cluster
  paths, SSH key names, or machine-specific operating instructions.
- Treat untrusted repositories and checkpoints as untrusted code/data; follow
  the boundaries in `SECURITY.md`.
- Isolate temporary files and clean them safely.
- Preserve unrelated user changes in mixed worktrees.

## Engineering expectations

- Python code must be typed.
- Public APIs require docstrings and tests.
- State paths and JSON output must be deterministic.
- Subprocess behavior must be tested.
- Keep examples small and reproducible.
- If the requested scope is insufficient, stop and report the exact blocker
  instead of broadening the change.
