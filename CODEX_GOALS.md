# Codex Goal Mode 执行序列

> 不要一次性让 Codex “完成整个项目”。官方 Goal Mode 更适合一个明确结果、一个停止条件和可执行验证。每个 Gate 使用单独 `/goal`，通过人工验收后再启动下一个。

---

## 初始化

把本目录中的以下文件放到新 Git 仓库根目录：

- `PLAN.md`
- `ACCEPTANCE.md`
- `AGENTS.md`
- `CODEX_GOALS.md`

启动 Codex 后，先执行：

```text
/init
```

若 `/init` 覆盖了本文件提供的 `AGENTS.md`，恢复本文件内容或合并必要规则。

---

## Gate 0 Goal

```text
/goal Execute Gate 0 from ACCEPTANCE.md only.

Outcome:
Produce a reproducible competitor/differentiation study against OrderLab/TrainCheck and a precise product contract for TrainParity.

Constraints:
- Do not implement the full library.
- Do not rely only on README claims; run experiments where feasible.
- Do not copy TrainCheck code.
- Build four stable fault fixtures: missing scheduler state, missing RNG state, gradient-accumulation mean-of-means, and sample duplication.
- Keep any TrainParity prototype throwaway and under 100 lines.
- If the differentiation is not structural, conclude STOP rather than forcing a project rationale.

Verification:
Run `python scripts/verify_gate.py 0`.
Generate `artifacts/gate_reports/gate_0.json` and `.md`.

Stopping condition:
Stop after the Gate 0 report and wait for human approval. Do not begin Gate 1.
```

验收人通过后发送：

```text
APPROVE GATE 0. Proceed only to Gate 1.
```

---

## Gate 1 Goal

```text
/goal Execute Gate 1 from ACCEPTANCE.md only.

Outcome:
Create the installable project skeleton, durable project instructions, CI, and two API prototypes, then select the lower-friction process-safe adapter design.

Constraints:
- No runtime LLM dependencies.
- No distributed support.
- No web UI or service.
- The selected simple adapter must be at most 30 logical lines.
- The resume case must be importable in a fresh Python process.
- Keep dependencies minimal and document every production dependency.

Verification:
Run lint, type-check, tests, wheel build, and `python scripts/verify_gate.py 1`.
Write Gate 1 JSON and Markdown reports.

Stopping condition:
Stop after Gate 1 verification and wait for human approval.
```

---

## Gate 2 Goal

```text
/goal Execute Gate 2 from ACCEPTANCE.md only.

Outcome:
Implement the deterministic state snapshot, canonicalization, comparison, and first-observed-divergence core.

Constraints:
- Use stable parameter-name-based optimizer paths.
- Exact and explicit tolerance policies must remain separate.
- Unsupported or ambiguous states must produce ABSTAIN, not false PASS.
- Do not implement resume orchestration yet beyond minimal test scaffolding.
- Do not add phase-level tracing unless required by Gate 2 tests.

Verification:
Run the full unit/contract suite and `python scripts/verify_gate.py 2`.
Meet the stated coverage and zero-false-positive criteria.

Stopping condition:
Stop after Gate 2 report and wait for human approval.
```

---

## Gate 3 Goal

```text
/goal Execute Gate 3 from ACCEPTANCE.md only.

Outcome:
Deliver the single-process/single-GPU Resume Equivalence MVP with real process termination, reload, self-consistency precheck, four-state results, and precise first-divergence reports.

Constraints:
- The resumed path must load in a new Python process.
- Do not emulate process exit inside one interpreter.
- Do not auto-guess numeric tolerance.
- Do not add DDP, Lightning, Transformers, a dashboard, or natural-language AI diagnosis.
- Preserve all fault fixtures.
- Distinguish training mismatch from runner failure.

Verification:
Run CPU verification and prepare the Slurm GPU command.
Run `python scripts/verify_gate.py 3`.
Meet all clean/fault/repeatability thresholds.

Stopping condition:
Stop after Gate 3 report. Explicitly recommend GO, REWORK, or STOP based on evidence, then wait for human approval.
```

---

## Gate 4 Goal

```text
/goal Execute Gate 4 from ACCEPTANCE.md only.

Outcome:
Integrate TrainParity with three small, license-compatible real PyTorch training recipes and measure actual adapter friction and diagnostic value.

Constraints:
- Avoid large datasets and long training.
- Do not patch the projects so heavily that the adapter metric becomes meaningless.
- Record adapter logical LOC automatically.
- Inject one realistic resume bug per project.
- Compare with a hand-written test and, where feasible, OrderLab/TrainCheck.

Verification:
Run `python scripts/verify_gate.py 4`.
All clean cases must pass; injected faults must fail; median adapter LOC must meet the threshold.

Stopping condition:
Stop after Gate 4 and recommend GO or STOP. Do not start accumulation work without approval.
```

---

## Gate 5 Goal

```text
/goal Execute Gate 5 from ACCEPTANCE.md only.

Outcome:
Add full-batch versus gradient-accumulation equivalence as a second check using the existing core.

Constraints:
- Require explicit batch splitting and comparison policy.
- Do not claim BatchNorm/dropout/global-batch losses are always equivalent.
- Compare loss accounting, gradients, optimizer state, and parameter update.
- Do not build a trainer.
- Preserve clean and fault fixtures.

Verification:
Run `python scripts/verify_gate.py 5` and all existing tests.
Meet all fault-detection and clean-case criteria.

Stopping condition:
Stop after Gate 5 report and wait for approval.
```

---

## Gate 6 Goal

```text
/goal Execute Gate 6 from ACCEPTANCE.md only.

Outcome:
Add sample-coverage policies with exact missing/duplicate/rank-overlap evidence.

Constraints:
- Require a stable sample-ID extractor.
- Keep the feature focused on coverage auditing, not distributed training orchestration.
- Prefer CPU multiprocessing/sampler simulation for tests.
- Do not require sample content logging.
- Do not add a dashboard.

Verification:
Run `python scripts/verify_gate.py 6` and all prior tests.

Stopping condition:
Stop after Gate 6 report and wait for approval.
```

---

## Gate 7 Goal

```text
/goal Execute Gate 7 from ACCEPTANCE.md only.

Outcome:
Prepare a public release candidate with a verified package, honest documentation, stable examples, CI, licensing, and release artifacts.

Constraints:
- Recheck package and repository name availability.
- Clearly distinguish TrainParity from OrderLab/TrainCheck.
- Do not publish, push tags, or create remote resources without explicit human authorization.
- Do not expand scope during release preparation.
- Keep all limitations visible in README.

Verification:
Run `make release-check` and `python scripts/verify_gate.py 7`.

Stopping condition:
Stop with a release-candidate report and wait for explicit publication approval.
```

---

## 建议的人类验收回复格式

通过：

```text
APPROVE GATE N.
Evidence reviewed:
- ...
Proceed only to Gate N+1.
```

返工：

```text
REWORK GATE N.
Failed acceptance items:
1. ...
2. ...
Do not start the next Gate.
```

终止：

```text
STOP PROJECT.
Reason:
...
Preserve the repository and produce a final lessons-learned report only.
```
