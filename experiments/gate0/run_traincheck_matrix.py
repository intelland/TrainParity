"""Run TrainCheck 0.1.2 as an isolated black-box Gate 0 experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

import torch

CASES = (
    "missing_scheduler_state",
    "missing_rng_state",
    "mean_of_means",
    "sample_duplication",
)
PHASE_TIMEOUT_SECONDS = 240


def _tail(text: str, limit: int = 3_000) -> str:
    return text[-limit:]


def _run(
    phase: str,
    command: list[str],
    cwd: Path,
    log_dir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=PHASE_TIMEOUT_SECONDS,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as error:
        returncode = 124
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        timed_out = True
    duration = round(time.perf_counter() - started, 3)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{phase}.log"
    log_path.write_text(
        f"command={shlex.join(command)}\nreturncode={returncode}\n"
        f"duration_seconds={duration}\ntimed_out={timed_out}\n"
        f"\n[stdout]\n{stdout}\n[stderr]\n{stderr}",
        encoding="utf-8",
    )
    return {
        "command": shlex.join(command),
        "duration_seconds": duration,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "log_path": str(log_path),
    }


def _failed_count(check_phase: dict[str, Any]) -> int | None:
    combined = check_phase["stdout_tail"] + "\n" + check_phase["stderr_tail"]
    match = re.search(r"Total failed invariants:\s*(\d+)", combined)
    return int(match.group(1)) if match else None


def _failure_evidence(check_dir: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for path in sorted(check_dir.rglob("failed.log")):
        content = path.read_text(encoding="utf-8", errors="replace")
        position = 0
        while position < len(content):
            while position < len(content) and content[position].isspace():
                position += 1
            if position == len(content):
                break
            record, position = decoder.raw_decode(content, position)
            invariant = record.get("invariant", {})
            trace = record.get("trace", [])
            event = trace[-1] if trace else {}
            evidence.append(
                {
                    "description": invariant.get("text_description"),
                    "relation": invariant.get("relation"),
                    "step": event.get("meta_vars.step"),
                    "stage": event.get("meta_vars.stage"),
                    "function": event.get("function"),
                }
            )
    return evidence


def _signature(evidence: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(evidence.get(key) for key in ("description", "relation", "step", "stage", "function"))


def _fault_specific(
    control: list[dict[str, Any]], fault: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    remaining = Counter(_signature(item) for item in fault)
    remaining.subtract(_signature(item) for item in control)
    selected: list[dict[str, Any]] = []
    for item in fault:
        signature = _signature(item)
        if remaining[signature] > 0:
            selected.append(item)
            remaining[signature] -= 1
    return selected


def _safe_reset(runtime_root: Path, project_root: Path) -> None:
    runtime_root = runtime_root.resolve()
    allowed_root = (project_root / "outputs" / "gate0").resolve()
    if runtime_root != allowed_root and allowed_root not in runtime_root.parents:
        raise ValueError(f"runtime root escapes Gate 0 outputs: {runtime_root}")
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True)


def main() -> None:
    """Execute four complete reference/infer/target/check workflows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
    repository = Path(__file__).resolve().parents[2]
    entries = repository / "experiments" / "gate0" / "competitor_entries"
    runtime_root = args.runtime_root.resolve()
    _safe_reset(runtime_root, project_root)
    binary_root = Path(sys.executable).parent
    collect = str(binary_root / "traincheck-collect")
    infer = str(binary_root / "traincheck-infer")
    check = str(binary_root / "traincheck-check")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["NUMBA_CACHE_DIR"] = str(project_root / "caches" / "numba")
    env["TMPDIR"] = str(project_root / "tmp")
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for case in CASES:
        case_root = runtime_root / case
        log_dir = case_root / "logs"
        reference_trace = case_root / "reference_trace"
        control_trace = case_root / "control_trace"
        fault_trace = case_root / "fault_trace"
        invariants = case_root / "invariants.json"
        control_check = case_root / "control_check"
        fault_check = case_root / "fault_check"
        phases: dict[str, dict[str, Any]] = {}
        phases["collect_reference"] = _run(
            "collect_reference",
            [collect, "--pyscript", str(entries / f"{case}_clean.py"), "--output-dir", str(reference_trace), "--models-to-track", "model"],
            repository,
            log_dir,
            env,
        )
        if phases["collect_reference"]["returncode"] == 0:
            phases["infer"] = _run(
                "infer",
                [infer, "--trace-folders", str(reference_trace), "--output", str(invariants), "--backend", "pandas"],
                repository,
                log_dir,
                env,
            )
        if phases.get("infer", {}).get("returncode") == 0:
            phases["collect_control"] = _run(
                "collect_control",
                [collect, "--pyscript", str(entries / f"{case}_clean.py"), "--invariants", str(invariants), "--output-dir", str(control_trace), "--models-to-track", "model"],
                repository,
                log_dir,
                env,
            )
        if phases.get("collect_control", {}).get("returncode") == 0:
            phases["check_control"] = _run(
                "check_control",
                [check, "--trace-folders", str(control_trace), "--invariants", str(invariants), "--backend", "pandas", "--output-dir", str(control_check), "--no-html-report"],
                repository,
                log_dir,
                env,
            )
        if phases.get("check_control", {}).get("returncode") == 0:
            phases["collect_fault"] = _run(
                "collect_fault",
                [collect, "--pyscript", str(entries / f"{case}_fault.py"), "--invariants", str(invariants), "--output-dir", str(fault_trace), "--models-to-track", "model"],
                repository,
                log_dir,
                env,
            )
        if phases.get("collect_fault", {}).get("returncode") == 0:
            phases["check_fault"] = _run(
                "check_fault",
                [check, "--trace-folders", str(fault_trace), "--invariants", str(invariants), "--backend", "pandas", "--output-dir", str(fault_check), "--no-html-report"],
                repository,
                log_dir,
                env,
            )
        complete = set(phases) == {"collect_reference", "infer", "collect_control", "check_control", "collect_fault", "check_fault"} and all(
            phase["returncode"] == 0 for phase in phases.values()
        )
        control_failures = _failed_count(phases["check_control"]) if complete else None
        fault_failures = _failed_count(phases["check_fault"]) if complete else None
        control_evidence = _failure_evidence(control_check) if complete else []
        fault_evidence = _failure_evidence(fault_check) if complete else []
        fault_specific = _fault_specific(control_evidence, fault_evidence)
        results.append(
            {
                "case": case,
                "status": "EXECUTED" if complete else "BLOCKED",
                "detected": bool(fault_specific),
                "control_failed_invariants": control_failures,
                "fault_failed_invariants": fault_failures,
                "fault_specific_violation_count": len(fault_specific),
                "control_evidence": control_evidence,
                "fault_evidence": fault_evidence,
                "fault_specific_evidence": fault_specific,
                "phases": phases,
            }
        )
    payload = {
        "schema_version": 1,
        "experiment_boundary": {
            "competitor": "traincheck==0.1.2 from PyPI",
            "method": "black-box CLI only; no TrainCheck source copied",
            "runtime_root": str(runtime_root),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "traincheck": importlib.metadata.version("traincheck"),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for result in results:
        duration = round(
            sum(phase["duration_seconds"] for phase in result["phases"].values()), 3
        )
        print(
            f"{result['case']}: status={result['status']} "
            f"detected={result['detected']} "
            f"control_failed={result['control_failed_invariants']} "
            f"fault_failed={result['fault_failed_invariants']} "
            f"fault_specific={result['fault_specific_violation_count']} "
            f"duration_seconds={duration}"
        )
    print(f"summary={args.output}")


if __name__ == "__main__":
    main()
