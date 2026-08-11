# AGENTS.md — TrainParity

## Mission

Build a small, deterministic PyTorch differential-testing library that verifies user-declared training equivalence and locates the first observable divergence.

Read, in order:

1. `docs/development/PLAN.md`
2. `docs/development/ACCEPTANCE.md`
3. `docs/development/STATUS.md`
4. `docs/development/DECISIONS.md`
5. `docs/development/CODEX_GOALS.md`

## Hard boundaries

- Do not call any LLM at TrainParity runtime.
- Do not build a web app, dashboard, service, registry, leaderboard, database, or agent wrapper.
- Do not claim support for arbitrary Python training scripts.
- Do not implement DDP/FSDP/DeepSpeed/Lightning/Transformers adapters unless a later Gate explicitly authorizes it.
- Do not copy code from OrderLab/TrainCheck.
- Do not weaken predicates, remove fault fixtures, or silently loosen tolerances to make tests pass.
- Do not proceed to the next Gate without explicit human approval.
- Treat `ABSTAIN`, `ERROR`, `FAIL`, and `PASS` as distinct outcomes.
- Report first *observed* divergence, not inferred root cause.
- Keep production dependencies minimal.

## Work protocol

For the active Gate:

1. Update `docs/development/STATUS.md` with the Gate objective, constraints, and verification commands.
2. Implement only that Gate.
3. Add or update tests before declaring completion.
4. Run lint, type-check, focused tests, full tests, and the Gate verifier.
5. Create:
   - `artifacts/gate_reports/gate_<N>.json`
   - `artifacts/gate_reports/gate_<N>.md`
6. Summarize:
   - what changed;
   - what was verified;
   - remaining limitations;
   - exact commands run;
   - whether any criterion is blocked.
7. Stop and request human acceptance.

If blocked, write a precise `BLOCKED` result. Never fabricate a pass.

## Standard verification

Prefer repository commands exposed through `Makefile`:

```bash
make lint
make typecheck
make test
python scripts/verify_gate.py <N>
```

Create Git checkpoints before and after each Gate. Avoid concurrent write access from multiple agents. Read-only review subagents are allowed.

## Engineering expectations

- Python code must be typed.
- Public APIs require docstrings and tests.
- State paths and JSON output must be deterministic.
- Subprocess behavior must be tested.
- Temporary files must be isolated and cleaned safely.
- Untrusted external repositories must not be executed without an explicit experiment boundary.
- Record new dependencies and architectural decisions in `docs/development/DECISIONS.md`.
- Keep examples tiny and reproducible.
