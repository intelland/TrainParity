"""Run TrainCheck 0.1.2 as an isolated black-box Gate 0 experiment."""

from __future__ import annotations

import argparse
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


def _tail(text: str, limit: int = 12_000) -> str:
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


def _failure_evidence(check_dir: Path) -> list[str]:
    evidence: list[str] = []
    for path in sorted(check_dir.rglob("failed.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped:
                evidence.append(stripped[:1000])
            if len(evidence) == 5:
                return evidence
    return evidence


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
        target_trace = case_root / "target_trace"
        invariants = case_root / "invariants.json"
        check_dir = case_root / "check"
        phases: dict[str, dict[str, Any]] = {}
        phases["collect_reference"] = _run(
            "collect_reference",
            [collect, "--pyscript", str(entries / f"{case}_clean.py"), "--copy-all-files", "--output-dir", str(reference_trace), "--models-to-track", "model"],
            repository,
            log_dir,
            env,
        )
        if phases["collect_reference"]["returncode"] == 0:
            phases["infer"] = _run(
                "infer",
                [infer, "--trace-folders", str(reference_trace), "--output", str(invariants), "--backend", "dict"],
                repository,
                log_dir,
                env,
            )
        if phases.get("infer", {}).get("returncode") == 0:
            phases["collect_target"] = _run(
                "collect_target",
                [collect, "--pyscript", str(entries / f"{case}_fault.py"), "--copy-all-files", "--invariants", str(invariants), "--output-dir", str(target_trace), "--models-to-track", "model"],
                repository,
                log_dir,
                env,
            )
        if phases.get("collect_target", {}).get("returncode") == 0:
            phases["check"] = _run(
                "check",
                [check, "--trace-folders", str(target_trace), "--invariants", str(invariants), "--backend", "dict", "--output-dir", str(check_dir), "--no-html-report"],
                repository,
                log_dir,
                env,
            )
        complete = set(phases) == {"collect_reference", "infer", "collect_target", "check"} and all(
            phase["returncode"] == 0 for phase in phases.values()
        )
        failures = _failed_count(phases["check"]) if complete else None
        results.append(
            {
                "case": case,
                "status": "EXECUTED" if complete else "BLOCKED",
                "detected": failures is not None and failures > 0,
                "failed_invariants": failures,
                "failure_evidence": _failure_evidence(check_dir) if complete else [],
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
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
