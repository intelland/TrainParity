"""Machine verification entry point for TrainParity acceptance gates."""

from __future__ import annotations

import argparse
import json
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prototype_lines(path: Path) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    logical = sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in lines)
    return len(lines), logical


def _write_reports(root: Path, report: dict[str, Any]) -> None:
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
        _write_reports(root, report)
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
    _write_reports(root, report)
    return report


def main() -> None:
    """Parse the Gate number, verify it, and return a process status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", type=int)
    args = parser.parse_args()
    if args.gate != 0:
        raise SystemExit(f"gate {args.gate} is not implemented")
    root = Path(__file__).resolve().parents[1]
    report = verify_gate_0(root)
    print(json.dumps({"gate": 0, "status": report["status"], "recommendation": report["recommendation"]}, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
