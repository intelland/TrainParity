"""Machine verification entry point for TrainParity acceptance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

EXPECTED_CASES = {
    "missing_scheduler_state": (2, "optimizer.lr"),
    "missing_rng_state": (2, "rng.torch"),
    "mean_of_means": (0, "gradient.model.weight"),
    "sample_duplication": (3, "batch.sample_ids.0"),
}
EXPECTED_PHASES = {
    "collect_reference",
    "infer",
    "collect_control",
    "check_control",
    "collect_fault",
    "check_fault",
}
GATE_0_EVIDENCE_HASHES = {
    "artifacts/gate_reports/gate_0.json": "3f72383d025304a8559773ac961b8fae0e1e7f0c21adaa6eb3bae07f2684dcc2",
    "artifacts/gate_reports/gate_0.md": "29f5fd42ee9a7f51126546e3dc02a558c1925bd7e1f84702fd12b84663eb53ad",
    "experiments/gate0/recorded/fault_matrix.json": "3006acba2857040d2e5b1851886b3abd052e3601fcd26812fd6c6b8e5d980a81",
    "experiments/gate0/recorded/traincheck_summary.json": "cd6d57bab81b4a3b31c64ee1942e61d673c868155883fc3ec85e174f9acc4df7",
}
GATE_1_EVIDENCE_HASHES = {
    "artifacts/gate_reports/gate_1.json": "5267ef045f8163a19a2ab71bfb483440e9778f9aec50d24c80dafab8c823d9ff",
    "artifacts/gate_reports/gate_1.md": "998443c965f919ff1b4b7aacef1a088906c5b7193fb799b32b0c82ee3b4ccf00",
    "experiments/gate1/recorded/adapter_evaluation.json": "a9e9618cea4ca0c1494c64a0f059eff447232f781a18e89ad7bafdb212714054",
}
EXPECTED_GATE_2_FAULTS = {
    "nested_value": "extra.nested.items[0].value",
    "tensor_shape": "extra.tensor",
    "tensor_dtype": "extra.tensor",
    "tensor_value": "extra.tensor",
    "nan": "extra",
    "inf": "extra",
    "empty_tensor": "extra",
    "device_metadata": "extra",
    "parameter_group_order": "optimizer.param_groups[0].lr",
    "sgd_momentum": "optimizer.state.weight.momentum_buffer",
    "adam_exp_avg": "optimizer.state.weight.exp_avg",
    "missing_key": "extra.a",
    "extra_key": "extra.b",
    "same_value_different_path": "extra.a",
    "user_state_dict": "extra.ema.shadow.weight",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prototype_lines(path: Path) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    logical = sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in lines)
    return len(lines), logical


def _write_gate_0_reports(root: Path, report: dict[str, Any]) -> None:
    report_dir = root / "artifacts" / "gate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "gate_0.json"
    md_path = report_dir / "gate_0.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    criteria_lines = "\n".join(
        f"- [{'x' if item['passed'] else ' '}] {item['name']}: {item['evidence']}"
        for item in report["criteria"]
    )
    fault_lines = "\n".join(
        f"- `{item['case']}`: step {item['step']}, `{item['path']}`"
        for item in report["metrics"]["prototype_first_divergences"]
    )
    competitor_lines = "\n".join(
        f"- `{item['case']}`: control={item['control_failed']}, "
        f"fault={item['fault_failed']}, specific={item['fault_specific']}, "
        f"detected={item['detected']}"
        for item in report["metrics"]["traincheck_results"]
    )
    commands = "\n".join(f"- `{command}`" for command in report["commands"])
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    markdown = f"""# Gate 0 report

## Outcome

**{report['status']} — recommendation: {report['recommendation']}**

{report['summary']}

## Acceptance criteria

{criteria_lines}

## TrainParity prototype

{fault_lines}

Prototype size: {report['metrics']['prototype_physical_lines']} physical lines,
{report['metrics']['prototype_logical_lines']} nonblank/noncomment lines.

## TrainCheck 0.1.2 with clean controls

{competitor_lines}

## Exact commands

{commands}

## Remaining limitations

{limitations}
"""
    md_path.write_text(markdown, encoding="utf-8")


def verify_gate_0(root: Path) -> dict[str, Any]:
    """Verify Gate 0 artifacts and return a machine-readable report."""
    faults_path = root / "experiments" / "gate0" / "recorded" / "fault_matrix.json"
    traincheck_path = root / "experiments" / "gate0" / "recorded" / "traincheck_summary.json"
    prototype_path = root / "experiments" / "gate0" / "ab_prototype.py"
    competitor_doc = root / "docs" / "COMPETITOR_ANALYSIS.md"
    contract_doc = root / "docs" / "PRODUCT_CONTRACT.md"
    required = [faults_path, traincheck_path, prototype_path, competitor_doc, contract_doc]
    criteria: list[dict[str, Any]] = []

    def criterion(name: str, passed: bool, evidence: str) -> None:
        criteria.append({"name": name, "passed": bool(passed), "evidence": evidence})

    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    criterion("required artifacts", not missing, "present" if not missing else f"missing: {missing}")
    if missing:
        report = {
            "schema_version": 1,
            "gate": 0,
            "status": "BLOCKED",
            "recommendation": "REWORK",
            "summary": "Gate 0 required artifacts are missing.",
            "criteria": criteria,
            "metrics": {"prototype_first_divergences": [], "traincheck_results": [], "prototype_physical_lines": 0, "prototype_logical_lines": 0},
            "commands": ["python scripts/verify_gate.py 0"],
            "limitations": ["Verification stopped before experiment records could be loaded."],
        }
        _write_gate_0_reports(root, report)
        return report

    faults = _load_json(faults_path)
    traincheck = _load_json(traincheck_path)
    fault_results = {item["case"]: item for item in faults.get("results", [])}
    traincheck_results = {item["case"]: item for item in traincheck.get("results", [])}
    criterion(
        "fault fixture inventory",
        set(fault_results) == set(EXPECTED_CASES),
        f"observed={sorted(fault_results)}",
    )
    fixture_errors = []
    prototype_metrics = []
    for case, (step, path) in EXPECTED_CASES.items():
        result = fault_results.get(case, {})
        divergence = result.get("first_divergence") or {}
        observed = (divergence.get("step"), divergence.get("path"))
        if result.get("outcome") != "FAIL" or observed != (step, path):
            fixture_errors.append(f"{case}: {observed}")
        prototype_metrics.append({"case": case, "step": observed[0], "path": observed[1]})
    criterion("stable expected first divergences", not fixture_errors, "all four exact" if not fixture_errors else "; ".join(fixture_errors))
    criterion(
        "three-run repeatability",
        faults.get("stable") is True and faults.get("repeat_count") == 3,
        f"stable={faults.get('stable')} repeats={faults.get('repeat_count')}",
    )
    physical_lines, logical_lines = _prototype_lines(prototype_path)
    criterion("prototype size", physical_lines < 100, f"physical={physical_lines}, logical={logical_lines}")
    criterion(
        "competitor inventory",
        set(traincheck_results) == set(EXPECTED_CASES),
        f"observed={sorted(traincheck_results)}",
    )
    phase_errors = []
    competitor_metrics = []
    for case in EXPECTED_CASES:
        result = traincheck_results.get(case, {})
        phases = result.get("phases", {})
        bad_phases = sorted(
            phase for phase in EXPECTED_PHASES if phases.get(phase, {}).get("returncode") != 0
        )
        if result.get("status") != "EXECUTED" or set(phases) != EXPECTED_PHASES or bad_phases:
            phase_errors.append(f"{case}: {bad_phases or 'phase inventory mismatch'}")
        competitor_metrics.append(
            {
                "case": case,
                "control_failed": result.get("control_failed_invariants"),
                "fault_failed": result.get("fault_failed_invariants"),
                "fault_specific": result.get("fault_specific_violation_count"),
                "detected": result.get("detected"),
            }
        )
    criterion("TrainCheck black-box execution", not phase_errors, "24/24 phases exited zero" if not phase_errors else "; ".join(phase_errors))
    control_present = all(
        result.get("control_failed_invariants") is not None
        and isinstance(result.get("fault_specific_evidence"), list)
        for result in traincheck_results.values()
    )
    criterion("clean-control correction", control_present, "control and fault signatures recorded")
    structural_cases = sorted(
        case
        for case in EXPECTED_CASES
        if fault_results.get(case, {}).get("outcome") == "FAIL"
        and not traincheck_results.get(case, {}).get("detected", False)
    )
    criterion(
        "structural differentiation threshold",
        len(structural_cases) >= 2,
        f"prototype-only precise detections={structural_cases}",
    )
    competitor_text = competitor_doc.read_text(encoding="utf-8")
    competitor_text_lower = competitor_text.lower()
    contract_text = contract_doc.read_text(encoding="utf-8")
    criterion(
        "competitor sources and limitations",
        "orderlab/traincheck" in competitor_text_lower
        and "clean control" in competitor_text_lower
        and "threats to validity" in competitor_text_lower
        and "pypi" in competitor_text_lower,
        "official sources, controls, and threats documented",
    )
    criterion(
        "four-state product contract",
        all(f"`{state}`" in contract_text for state in ("PASS", "FAIL", "ABSTAIN", "ERROR"))
        and "first observed divergence" in contract_text,
        "PASS/FAIL/ABSTAIN/ERROR and first-observed semantics documented",
    )
    passed = all(item["passed"] for item in criteria)
    report = {
        "schema_version": 1,
        "gate": 0,
        "status": "PASS" if passed else "BLOCKED",
        "recommendation": "GO" if passed else "REWORK",
        "summary": (
            "Controlled evidence supports an explicit A/B plus first-state-path product distinction. "
            "Human approval is required before Gate 1."
            if passed
            else "One or more Gate 0 machine criteria failed; do not proceed."
        ),
        "criteria": criteria,
        "metrics": {
            "prototype_physical_lines": physical_lines,
            "prototype_logical_lines": logical_lines,
            "prototype_first_divergences": prototype_metrics,
            "traincheck_environment": traincheck.get("environment"),
            "traincheck_results": competitor_metrics,
            "structural_difference_cases": structural_cases,
        },
        "commands": [
            "python -m experiments.gate0.run_fault_matrix --output $PROJECT_ROOT/outputs/gate0/recorded/fault_matrix.json",
            "python -m experiments.gate0.run_traincheck_matrix --runtime-root $PROJECT_ROOT/outputs/gate0/traincheck --output $PROJECT_ROOT/outputs/gate0/recorded/traincheck_summary.json",
            "python -m ruff check experiments/gate0 scripts/verify_gate.py",
            "python -m mypy --ignore-missing-imports --check-untyped-defs experiments/gate0 scripts/verify_gate.py",
            "python -m compileall -q experiments scripts",
            "python scripts/verify_gate.py 0",
        ],
        "limitations": [
            "Four tiny CPU fixtures do not establish real-repository adapter cost.",
            "TrainCheck was tested only at version 0.1.2 with PyTorch 2.13.0+cpu and the pandas backend.",
            "The throwaway prototype does not implement a true fresh-process resume runner.",
            "Gate 0 provides no production TrainParity API or arbitrary-script support.",
        ],
    }
    _write_gate_0_reports(root, report)
    return report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_gate_1_reports(root: Path, report: dict[str, Any]) -> None:
    report_dir = root / "artifacts" / "gate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "gate_1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    criteria = "\n".join(
        f"- [{'x' if item['passed'] else ' '}] {item['name']}: {item['evidence']}"
        for item in report["criteria"]
    )
    commands = "\n".join(f"- `{command}`" for command in report["commands"])
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    markdown = f"""# Gate 1 report

## Outcome

**{report['status']} — recommendation: {report['recommendation']}**

{report['summary']}

## Acceptance criteria

{criteria}

## API selection

Selected: `{report['metrics']['selected_api']}`.

Selected simple adapter: {report['metrics']['adapter_logical_lines']} logical lines.
Correct resume example: `{report['metrics']['correct_resume_outcome']}`.
Faulty resume example: `{report['metrics']['faulty_resume_outcome']}` at
`{report['metrics']['faulty_first_observed_divergence']}`.

## Exact commands

{commands}

## Remaining limitations

{limitations}
"""
    (report_dir / "gate_1.md").write_text(markdown, encoding="utf-8")


def verify_gate_1(root: Path) -> dict[str, Any]:
    """Verify the Gate 1 skeleton and API evaluation evidence."""
    evaluation_path = root / "experiments" / "gate1" / "recorded" / "adapter_evaluation.json"
    required = [
        root / "pyproject.toml",
        root / "Makefile",
        root / ".github" / "workflows" / "ci.yml",
        root / "src" / "trainparity" / "protocols.py",
        root / "src" / "trainparity" / "importing.py",
        root / "src" / "trainparity" / "examples" / "resume_cases.py",
        root / "docs" / "API_PROTOTYPES.md",
        root / "tests" / "test_importing.py",
        root / "tests" / "test_resume_examples.py",
        evaluation_path,
    ]
    criteria: list[dict[str, Any]] = []

    def criterion(name: str, passed: bool, evidence: str) -> None:
        criteria.append({"name": name, "passed": bool(passed), "evidence": evidence})

    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    criterion("installable skeleton and engineering files", not missing, "present" if not missing else f"missing: {missing}")
    if missing:
        report = {
            "schema_version": 1,
            "gate": 1,
            "status": "BLOCKED",
            "recommendation": "REWORK",
            "summary": "Gate 1 required artifacts are missing; do not begin Gate 2.",
            "criteria": criteria,
            "metrics": {
                "selected_api": None,
                "adapter_logical_lines": None,
                "correct_resume_outcome": None,
                "faulty_resume_outcome": None,
                "faulty_first_observed_divergence": None,
            },
            "commands": ["python scripts/verify_gate.py 1"],
            "limitations": ["Verification stopped before API evidence could be loaded."],
        }
        _write_gate_1_reports(root, report)
        return report

    evaluation = _load_json(evaluation_path)
    class_prototype = evaluation.get("prototypes", {}).get("class_protocol", {})
    callback_prototype = evaluation.get("prototypes", {}).get("factory_callbacks", {})
    cases = evaluation.get("resume_cases", {})
    correct = cases.get("correct", {})
    faulty = cases.get("missing_scheduler_state", {})
    adapter_lines = class_prototype.get("adapter_logical_lines")
    criterion(
        "two API prototypes evaluated",
        evaluation.get("selected_api") == "class_protocol"
        and class_prototype.get("type_surface") == "ResumeCase protocol"
        and callback_prototype.get("type_surface") == "ResumeCallbacks dataclass",
        "class/protocol selected after factory-plus-callback comparison",
    )
    criterion(
        "selected adapter size",
        isinstance(adapter_lines, int) and adapter_lines <= 30,
        f"logical_lines={adapter_lines}",
    )
    imported = class_prototype.get("fresh_process_import", {})
    criterion(
        "fresh-process import",
        class_prototype.get("process_safe") is True and imported.get("returncode") == 0,
        f"returncode={imported.get('returncode')}",
    )
    criterion(
        "no cloudpickle requirement",
        class_prototype.get("cloudpickle_required") is False
        and callback_prototype.get("cloudpickle_required") is False,
        "both prototypes use ordinary module imports",
    )
    criterion(
        "correct resume case",
        correct.get("outcome") == "PASS" and correct.get("first_observed_divergence") is None,
        f"outcome={correct.get('outcome')}",
    )
    criterion(
        "faulty resume case",
        faulty.get("outcome") == "FAIL"
        and isinstance(faulty.get("first_observed_divergence"), str),
        f"outcome={faulty.get('outcome')}, first observed={faulty.get('first_observed_divergence')}",
    )
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = project.get("dependencies", [])
    forbidden = ("openai", "langchain", "cloudpickle", "ray", "fastapi", "flask")
    criterion(
        "minimal documented production dependencies",
        dependencies == ["torch>=2.5"]
        and all(name not in dependency.lower() for dependency in dependencies for name in forbidden)
        and "sole production dependency is `torch>=2.5`"
        in (root / "README.md").read_text(encoding="utf-8"),
        f"dependencies={dependencies}",
    )
    wheels = sorted((root / "dist").glob("trainparity-*.whl"))
    criterion("wheel build", bool(wheels), f"wheels={[path.name for path in wheels]}")
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    criterion(
        "CI covers Gate 1 checks",
        all(command in workflow for command in ("make lint", "make typecheck", "make test", "make build", "verify_gate.py 1")),
        "lint, type-check, pytest, wheel, and verifier steps present",
    )
    changed_evidence = [
        relative
        for relative, digest in GATE_0_EVIDENCE_HASHES.items()
        if not (root / relative).is_file() or _sha256(root / relative) != digest
    ]
    criterion(
        "accepted Gate 0 evidence preserved",
        not changed_evidence,
        "hashes unchanged" if not changed_evidence else f"changed={changed_evidence}",
    )
    passed = all(item["passed"] for item in criteria)
    report = {
        "schema_version": 1,
        "gate": 1,
        "status": "PASS" if passed else "BLOCKED",
        "recommendation": "HUMAN_REVIEW" if passed else "REWORK",
        "summary": (
            "The installable Gate 1 skeleton and class/protocol adapter satisfy machine acceptance. "
            "Human approval is required before Gate 2."
            if passed
            else "One or more Gate 1 criteria failed; do not begin Gate 2."
        ),
        "criteria": criteria,
        "metrics": {
            "selected_api": evaluation.get("selected_api"),
            "selection_reasons": evaluation.get("selection_reasons"),
            "adapter_logical_lines": adapter_lines,
            "callback_factory_logical_lines": callback_prototype.get("factory_logical_lines"),
            "correct_resume_outcome": correct.get("outcome"),
            "faulty_resume_outcome": faulty.get("outcome"),
            "faulty_first_observed_divergence": faulty.get("first_observed_divergence"),
            "environment": evaluation.get("environment"),
            "wheels": [path.name for path in wheels],
            "production_dependencies": dependencies,
        },
        "commands": [
            "python -m experiments.gate1.run_adapter_evaluation --output $PROJECT_ROOT/outputs/gate1/adapter_evaluation.json",
            "make lint",
            "make typecheck",
            "make test",
            "make build",
            "python scripts/verify_gate.py 1",
            "git diff --check",
        ],
        "limitations": [
            "The direct resume probe is Gate 1 evidence, not a production runner or comparator.",
            "The tiny CPU cases do not establish compatibility with real training repositories.",
            "Only ordinary importable zero-argument classes are accepted; arbitrary scripts and local closures are unsupported.",
            "Distributed training, framework adapters, services, and runtime LLM/agent integration are out of scope.",
            "First observed divergence is not presented as root cause.",
        ],
    }
    _write_gate_1_reports(root, report)
    return report


def _write_gate_2_reports(root: Path, report: dict[str, Any]) -> None:
    report_dir = root / "artifacts" / "gate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "gate_2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    criteria = "\n".join(
        f"- [{'x' if item['passed'] else ' '}] {item['name']}: {item['evidence']}"
        for item in report["criteria"]
    )
    samples = "\n".join(
        f"- `{item['case']}` at `{item['path']}` ({item['reason']}): "
        f"{item['baseline']} -> {item['candidate']}"
        for item in report["metrics"]["sample_differences"]
    )
    commands = "\n".join(f"- `{command}`" for command in report["commands"])
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    markdown = f"""# Gate 2 report

## Outcome

**{report['status']} — recommendation: {report['recommendation']}**

{report['summary']}

## Acceptance criteria

{criteria}

## Representative first observed differences

{samples}

These are first observed divergences, not root-cause claims.

## Metrics

- Fault paths matched: {report['metrics']['faults_matched']}/{report['metrics']['fault_count']}
- Clean false positives: {report['metrics']['clean_false_positives']}
- Core coverage: {report['metrics']['core_coverage_percent']:.2f}%
- Ambiguous optimizer outcome: `{report['metrics']['ambiguous_optimizer_outcome']}`

## Exact commands

{commands}

## Remaining limitations

{limitations}
"""
    (report_dir / "gate_2.md").write_text(markdown, encoding="utf-8")


def verify_gate_2(root: Path) -> dict[str, Any]:
    """Verify Gate 2 snapshot/comparison evidence and write reports."""
    suite_path = root / "experiments" / "gate2" / "recorded" / "fault_suite.json"
    coverage_path = root / "experiments" / "gate2" / "recorded" / "coverage.json"
    ci_path = root / "experiments" / "gate2" / "recorded" / "ci_ae75212.json"
    required = [
        root / "src" / "trainparity" / "state.py",
        root / "src" / "trainparity" / "snapshot.py",
        root / "src" / "trainparity" / "optimizer_state.py",
        root / "src" / "trainparity" / "comparison.py",
        root / "docs" / "SNAPSHOT_CONTRACT.md",
        root / "tests" / "test_state.py",
        root / "tests" / "test_snapshot.py",
        root / "tests" / "test_optimizer_state.py",
        root / "tests" / "test_comparison.py",
        suite_path,
        coverage_path,
        ci_path,
    ]
    criteria: list[dict[str, Any]] = []

    def criterion(name: str, passed: bool, evidence: str) -> None:
        criteria.append({"name": name, "passed": bool(passed), "evidence": evidence})

    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    criterion(
        "Gate 2 implementation and evidence files",
        not missing,
        "present" if not missing else f"missing: {missing}",
    )
    if missing:
        report = {
            "schema_version": 1,
            "gate": 2,
            "status": "BLOCKED",
            "recommendation": "REWORK",
            "summary": "Gate 2 required artifacts are missing; do not begin Gate 3.",
            "criteria": criteria,
            "metrics": {
                "fault_count": 0,
                "faults_matched": 0,
                "clean_false_positives": 0,
                "core_coverage_percent": 0.0,
                "ambiguous_optimizer_outcome": None,
                "sample_differences": [],
            },
            "commands": ["python scripts/verify_gate.py 2"],
            "limitations": ["Verification stopped before required records could be loaded."],
        }
        _write_gate_2_reports(root, report)
        return report

    suite = _load_json(suite_path)
    coverage = _load_json(coverage_path)
    ci = _load_json(ci_path)
    faults = {item["case"]: item for item in suite.get("faults", [])}
    clean = {item["case"]: item for item in suite.get("clean", [])}
    criterion(
        "required fault inventory",
        set(faults) == set(EXPECTED_GATE_2_FAULTS),
        f"observed={sorted(faults)}",
    )
    mismatches = [
        name
        for name, expected in EXPECTED_GATE_2_FAULTS.items()
        if faults.get(name, {}).get("outcome") != "FAIL"
        or faults.get(name, {}).get("observed_path") != expected
    ]
    criterion(
        "fault suite expected stable paths",
        not mismatches,
        "15/15 exact paths" if not mismatches else f"mismatched={mismatches}",
    )
    false_positives = [name for name, item in clean.items() if item.get("outcome") != "PASS"]
    criterion(
        "clean suite zero false positives",
        set(clean) == set(EXPECTED_GATE_2_FAULTS) and not false_positives,
        "0/15" if not false_positives else f"false_positives={false_positives}",
    )
    detailed = all(
        item.get("difference", {}).get("reason")
        and item.get("difference", {}).get("baseline")
        and item.get("difference", {}).get("candidate")
        for item in faults.values()
    )
    criterion(
        "actionable non-binary difference reports",
        detailed,
        "path, reason, baseline, and candidate present for every fault",
    )
    policies = suite.get("policy_separation", {})
    criterion(
        "exact and tolerance policies remain separate",
        policies == {"exact": "FAIL", "tolerance": "PASS"},
        f"observed={policies}",
    )
    ambiguous = suite.get("ambiguous_optimizer", {})
    criterion(
        "ambiguous optimizer mapping abstains",
        ambiguous.get("outcome") == "ABSTAIN" and "aliases" in ambiguous.get("detail", ""),
        f"outcome={ambiguous.get('outcome')}, path={ambiguous.get('path')}",
    )
    optimizer_paths = {
        faults.get("sgd_momentum", {}).get("observed_path"),
        faults.get("adam_exp_avg", {}).get("observed_path"),
    }
    criterion(
        "optimizer paths use names",
        optimizer_paths
        == {
            "optimizer.state.weight.momentum_buffer",
            "optimizer.state.weight.exp_avg",
        },
        f"paths={sorted(str(path) for path in optimizer_paths)}",
    )
    criterion(
        "captured tensor state is alias-free",
        suite.get("tensor_alias_frozen") == "PASS",
        f"mutation probe={suite.get('tensor_alias_frozen')}",
    )
    coverage_percent = float(coverage.get("totals", {}).get("percent_covered", 0.0))
    criterion(
        "core module coverage",
        coverage_percent >= 90.0,
        f"percent_covered={coverage_percent:.2f}",
    )
    preserved_hashes = {**GATE_0_EVIDENCE_HASHES, **GATE_1_EVIDENCE_HASHES}
    changed_evidence = [
        relative
        for relative, digest in preserved_hashes.items()
        if not (root / relative).is_file() or _sha256(root / relative) != digest
    ]
    criterion(
        "accepted Gate 0 and Gate 1 evidence preserved",
        not changed_evidence,
        "hashes unchanged" if not changed_evidence else f"changed={changed_evidence}",
    )
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    criterion(
        "CI runs Gate 2 verifier",
        "verify_gate.py 2" in workflow and all(
            command in workflow for command in ("make lint", "make typecheck", "make test", "make build")
        ),
        "lint, type-check, tests, build, and Gate 2 verifier configured",
    )
    criterion(
        "Gate 1 hosted CI carry-forward confirmed",
        ci.get("commit") == "ae75212787f9cde54aae1391adced88650dc7ab3"
        and ci.get("conclusion") == "success",
        f"source={ci.get('source')}, conclusion={ci.get('conclusion')}",
    )
    passed = all(item["passed"] for item in criteria)
    sample_differences = [
        {
            "case": name,
            "path": faults[name]["difference"]["path"],
            "reason": faults[name]["difference"]["reason"],
            "baseline": faults[name]["difference"]["baseline"],
            "candidate": faults[name]["difference"]["candidate"],
        }
        for name in (
            "tensor_shape",
            "nan",
            "parameter_group_order",
            "sgd_momentum",
            "user_state_dict",
        )
    ]
    report = {
        "schema_version": 1,
        "gate": 2,
        "status": "PASS" if passed else "BLOCKED",
        "recommendation": "HUMAN_REVIEW" if passed else "REWORK",
        "summary": (
            "Gate 2 snapshot, canonicalization, and comparison satisfy machine acceptance. "
            "Human approval is required before Gate 3."
            if passed
            else "One or more Gate 2 criteria failed; do not begin Gate 3."
        ),
        "criteria": criteria,
        "metrics": {
            "fault_count": suite.get("fault_count"),
            "faults_matched": suite.get("faults_with_expected_path"),
            "clean_false_positives": suite.get("clean_false_positives"),
            "core_coverage_percent": coverage_percent,
            "ambiguous_optimizer_outcome": ambiguous.get("outcome"),
            "environment": suite.get("environment"),
            "sample_differences": sample_differences,
            "hosted_ci_confirmation": ci,
        },
        "commands": [
            "python -m experiments.gate2.run_fault_suite --output $PROJECT_ROOT/outputs/gate2/fault_suite.json",
            "make lint",
            "make typecheck",
            "make test",
            "make build",
            "python scripts/verify_gate.py 2",
            "git diff --check",
        ],
        "limitations": [
            "FullValueBackend is a correctness reference, not the only permitted future storage backend.",
            "Gate 2 compares one snapshot and does not implement trajectory or resume orchestration.",
            "Sparse and non-strided tensors currently produce ABSTAIN.",
            "CUDA metadata behavior is contract-tested without requiring a GPU allocation.",
            "First observed divergence is not presented as root cause.",
        ],
    }
    _write_gate_2_reports(root, report)
    return report


def main() -> None:
    """Parse the Gate number, verify it, and return a process status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.gate == 0:
        report = verify_gate_0(root)
    elif args.gate == 1:
        report = verify_gate_1(root)
    elif args.gate == 2:
        report = verify_gate_2(root)
    else:
        raise SystemExit(f"gate {args.gate} is not implemented")
    print(
        json.dumps(
            {"gate": args.gate, "status": report["status"], "recommendation": report["recommendation"]},
            sort_keys=True,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
