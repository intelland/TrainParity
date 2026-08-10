"""User-visible clean resume audit copied verbatim into a fresh upstream clone."""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import trainparity_adapter as adapter
import trainparity_project_glue as project

from trainparity.comparison import ExactComparison
from trainparity.serialization import encode_snapshot
from trainparity.snapshot import Snapshot
from trainparity.state import FrozenMapping, FullValueBackend


def _stable_keys(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, child in value.items():
            stable_key = str(key)
            if stable_key in converted:
                raise ValueError(f"checkpoint key collision after string conversion: {stable_key!r}")
            converted[stable_key] = _stable_keys(child)
        return converted
    if isinstance(value, list):
        return [_stable_keys(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stable_keys(item) for item in value)
    return value


def _peak_rss(path: Path) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Maximum resident set size" in line:
            return int(line.rsplit(":", 1)[1].strip())
    raise RuntimeError(f"missing peak RSS in {path}")


def _execute(
    checkout: Path,
    workspace: Path,
    name: str,
    end_step: int,
    resume: Path | None,
    timeout: float,
) -> dict[str, Any]:
    run_dir = workspace / name
    run_dir.mkdir(parents=True, exist_ok=True)
    command = project.command(checkout, run_dir, end_step, resume)
    time_path = run_dir / "time.txt"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    timed = ["/usr/bin/time", "-v", "-o", str(time_path), *command]
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            timed,
            cwd=checkout,
            env=os.environ,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{name} failed ({completed.returncode}): {tail}")
    pid_path = run_dir / "process.pid"
    return {
        "command": command,
        "elapsed_seconds": elapsed,
        "peak_rss_kib": _peak_rss(time_path),
        "process_pid": int(pid_path.read_text(encoding="utf-8")),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _capture(checkpoint: Path, step: int) -> tuple[Snapshot, float]:
    started = time.perf_counter()
    frozen = FullValueBackend().freeze(_stable_keys(adapter.normalize(checkpoint)))
    elapsed = time.perf_counter() - started
    if not isinstance(frozen, FrozenMapping):
        raise TypeError("normalized checkpoint root must be a mapping")
    return Snapshot(step, frozen), elapsed


def _serialize(snapshot: Snapshot, destination: Path) -> tuple[int, float]:
    started = time.perf_counter()
    payload = json.dumps(encode_snapshot(snapshot), sort_keys=True).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return len(payload), time.perf_counter() - started


def _tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def run(checkout: Path, workspace: Path, output: Path, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    setup_started = time.perf_counter()
    project.prepare(checkout, workspace / "data")
    setup_seconds = time.perf_counter() - setup_started

    normal_a = _execute(checkout, workspace, "normal_a", project.TOTAL_STEP, None, timeout)
    normal_b = _execute(checkout, workspace, "normal_b", project.TOTAL_STEP, None, timeout)
    split = _execute(checkout, workspace, "candidate_split", project.SPLIT_STEP, None, timeout)
    candidate_dir = workspace / "candidate_resume"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_input = adapter.checkpoint_path(candidate_dir)
    shutil.copy2(adapter.checkpoint_path(workspace / "candidate_split"), candidate_input)
    resumed = _execute(
        checkout,
        workspace,
        "candidate_resume",
        project.TOTAL_STEP,
        candidate_input,
        timeout,
    )

    snapshots: dict[str, Snapshot] = {}
    capture_seconds = 0.0
    serialization_seconds = 0.0
    snapshot_sizes: dict[str, int] = {}
    for name in ("normal_a", "normal_b", "candidate_resume"):
        snapshot, elapsed = _capture(
            adapter.checkpoint_path(workspace / name), project.TOTAL_STEP
        )
        snapshots[name] = snapshot
        capture_seconds += elapsed
        size, elapsed = _serialize(snapshot, workspace / "snapshots" / f"{name}.json")
        snapshot_sizes[name] = size
        serialization_seconds += elapsed

    comparison_started = time.perf_counter()
    self_differences = ExactComparison().compare_all(snapshots["normal_a"], snapshots["normal_b"])
    clean_differences = ExactComparison().compare_all(
        snapshots["normal_a"], snapshots["candidate_resume"]
    )
    comparison_seconds = time.perf_counter() - comparison_started
    total_seconds = time.perf_counter() - started
    processes = [normal_a, normal_b, split, resumed]
    result: dict[str, Any] = {
        "schema_version": 1,
        "project": adapter.NAME,
        "outcome": "PASS" if not self_differences and not clean_differences else "FAIL",
        "baseline_self_consistent": not self_differences,
        "clean_resume_equivalent": not clean_differences,
        "first_observed_divergence": None
        if not clean_differences
        else asdict(clean_differences[0]),
        "timing_seconds": {
            "setup": setup_seconds,
            "single_normal_run": normal_a["elapsed_seconds"],
            "second_baseline_run": normal_b["elapsed_seconds"],
            "baseline_self_consistency": normal_a["elapsed_seconds"]
            + normal_b["elapsed_seconds"],
            "candidate_save_exit": split["elapsed_seconds"],
            "candidate_new_process_load_resume": resumed["elapsed_seconds"],
            "candidate_save_exit_new_process_load_resume": split["elapsed_seconds"]
            + resumed["elapsed_seconds"],
            "snapshot_capture": capture_seconds,
            "serialization": serialization_seconds,
            "comparison": comparison_seconds,
            "total_wall": total_seconds,
            "end_to_end_multiplier": total_seconds / normal_a["elapsed_seconds"],
        },
        "peak_rss_kib": max(
            [item["peak_rss_kib"] for item in processes]
            + [resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]
        ),
        "snapshot_sizes_bytes": snapshot_sizes,
        "processes": processes,
        "fresh_resume_processes_distinct": split["process_pid"] != resumed["process_pid"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["total_artifact_size_bytes"] = _tree_size(workspace)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    result = run(args.checkout.resolve(), args.workspace.resolve(), args.output.resolve(), args.timeout)
    print(json.dumps({"outcome": result["outcome"], "project": result["project"]}))
    if result["outcome"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
