"""Run Gate 4 against three pinned, unmodified external repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trainparity.comparison import Difference, ExactComparison
from trainparity.serialization import encode_snapshot
from trainparity.snapshot import Snapshot
from trainparity.state import FrozenMapping, FullValueBackend

from experiments.gate4.adapters import IgniteMnistAdapter, ImageNetAdapter, NanoGptAdapter
from experiments.gate4.handwritten import final_state_equal
from experiments.gate4.models import ExternalProjectAdapter

ADAPTERS: tuple[ExternalProjectAdapter, ...] = (ImageNetAdapter(), NanoGptAdapter(), IgniteMnistAdapter())
DRIVER_FILES = {
    "pytorch_examples_imagenet": Path("experiments/gate4/drivers/imagenet.py"),
    "nanogpt": Path("experiments/gate4/drivers/nanogpt.py"),
    "ignite_mnist_engine": Path("experiments/gate4/drivers/ignite_mnist.py"),
}


def _logical_lines(path: Path, *, marked: bool = False) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if marked:
        start = lines.index("# ADAPTER LOGIC START") + 1
        end = lines.index("# ADAPTER LOGIC END")
        lines = lines[start:end]
    return sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in lines)


def _adapter_path(adapter: ExternalProjectAdapter) -> Path:
    module = sys.modules[type(adapter).__module__]
    return Path(str(module.__file__)).resolve()


def _run(
    command: list[str],
    *,
    run_dir: Path,
    environment: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "time.txt"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    actual = [sys.executable if value == "python" else value for value in command]
    timed = ["/usr/bin/time", "-v", "-o", str(metrics_path), *actual]
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(timed, stdout=stdout, stderr=stderr, env=environment, timeout=timeout)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"external command failed ({completed.returncode}): {tail}")
    peak_kib = 0
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if "Maximum resident set size" in line:
            peak_kib = int(line.rsplit(":", 1)[1].strip())
    return {
        "command": actual,
        "elapsed_seconds": elapsed,
        "peak_rss_kib": peak_kib,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _snapshot(adapter: ExternalProjectAdapter, path: Path, step: int) -> tuple[Snapshot, int]:
    frozen = FullValueBackend().freeze(adapter.normalize_checkpoint(path))
    if not isinstance(frozen, FrozenMapping):
        raise TypeError("normalized checkpoint root must be a mapping")
    snapshot = Snapshot(step, frozen)
    size = len(json.dumps(encode_snapshot(snapshot), sort_keys=True).encode("utf-8"))
    return snapshot, size


def _differences(left: Snapshot, right: Snapshot) -> tuple[Difference, ...]:
    return ExactComparison().compare_all(left, right)


def _repo_metadata(adapter: ExternalProjectAdapter, checkout: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != adapter.commit:
        raise RuntimeError(f"{adapter.name} expected {adapter.commit}, observed {commit}")
    license_path = checkout / "LICENSE"
    changed = subprocess.check_output(
        ["git", "-C", str(checkout), "diff", "--numstat", adapter.commit, "--"], text=True
    ).strip()
    modified_lines = 0
    for line in changed.splitlines():
        added, deleted, _ = line.split("\t", 2)
        modified_lines += (0 if added == "-" else int(added)) + (0 if deleted == "-" else int(deleted))
    return {
        "repository": adapter.repository,
        "commit": commit,
        "license": adapter.license_id,
        "license_file": str(license_path),
        "license_sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
        "upstream_modified_loc": modified_lines,
    }


def _first_divergence(
    baseline: dict[int, Snapshot], candidate: dict[int, Snapshot]
) -> tuple[int | None, tuple[Difference, ...]]:
    for step in sorted(candidate):
        differences = _differences(baseline[step], candidate[step])
        if differences:
            return step, differences
    return None, ()


def _project(
    adapter: ExternalProjectAdapter,
    *,
    root: Path,
    checkout: Path,
    environment: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    data_root = root / "data"
    executions: list[dict[str, Any]] = []
    executions.append(_run(adapter.prepare_command(checkout, data_root), run_dir=root / "prepare", environment=environment, timeout=timeout))
    baseline_paths: dict[int, Path] = {}
    baseline_snapshots: dict[int, Snapshot] = {}
    snapshot_sizes: list[int] = []
    for step in range(adapter.split_step, adapter.total_step + 1):
        run_dir = root / f"baseline_{step}"
        executions.append(_run(adapter.run_command(checkout, data_root, run_dir, step, None), run_dir=run_dir, environment=environment, timeout=timeout))
        checkpoint = adapter.checkpoint_path(run_dir)
        baseline_paths[step] = checkpoint
        baseline_snapshots[step], size = _snapshot(adapter, checkpoint, step)
        snapshot_sizes.append(size)

    clean_snapshots: dict[int, Snapshot] = {}
    fault_snapshots: dict[int, Snapshot] = {}
    comparison_seconds = 0.0
    for kind, target in (("clean", clean_snapshots), ("fault", fault_snapshots)):
        for step in range(adapter.split_step + 1, adapter.total_step + 1):
            run_dir = root / f"{kind}_{step}"
            destination = adapter.checkpoint_path(run_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_paths[adapter.split_step], destination)
            if kind == "fault":
                adapter.inject_fault(destination)
            executions.append(_run(adapter.run_command(checkout, data_root, run_dir, step, destination), run_dir=run_dir, environment=environment, timeout=timeout))
            target[step], size = _snapshot(adapter, adapter.checkpoint_path(run_dir), step)
            snapshot_sizes.append(size)

    started = time.perf_counter()
    clean_step, clean_differences = _first_divergence(baseline_snapshots, clean_snapshots)
    fault_step, fault_differences = _first_divergence(baseline_snapshots, fault_snapshots)
    comparison_seconds += time.perf_counter() - started
    handwritten_clean = final_state_equal(
        adapter.handwritten_state(baseline_paths[adapter.total_step]),
        adapter.handwritten_state(adapter.checkpoint_path(root / f"clean_{adapter.total_step}")),
    )
    handwritten_fault = final_state_equal(
        adapter.handwritten_state(baseline_paths[adapter.total_step]),
        adapter.handwritten_state(adapter.checkpoint_path(root / f"fault_{adapter.total_step}")),
    )
    adapter_path = _adapter_path(adapter)
    driver_path = Path(__file__).resolve().parents[2] / DRIVER_FILES[adapter.name]
    adapter_loc = _logical_lines(adapter_path, marked=True)
    glue_loc = _logical_lines(driver_path)
    upstream_seconds = sum(item["elapsed_seconds"] for item in executions)
    return {
        "name": adapter.name,
        "structure": adapter.structure,
        "fault": adapter.fault_name,
        "repository": _repo_metadata(adapter, checkout),
        "checkpoint_implementation": "original upstream save and load path",
        "clean": {"outcome": "PASS" if clean_step is None else "FAIL", "first_divergent_step": clean_step},
        "fault_result": {
            "outcome": "FAIL" if fault_step is not None else "PASS",
            "first_divergent_step": fault_step,
            "primary_difference": None if not fault_differences else asdict(fault_differences[0]),
            "all_differences": [asdict(item) for item in fault_differences],
        },
        "handwritten": {
            "clean_outcome": "PASS" if handwritten_clean else "FAIL",
            "fault_outcome": "PASS" if handwritten_fault else "FAIL",
            "diagnostic": "final model states are equal" if handwritten_fault else "final model states differ",
        },
        "loc": {
            "adapter_logical": adapter_loc,
            "supporting_glue_logical": glue_loc,
            "upstream_modified": 0,
            "total_new_project_integration": adapter_loc + glue_loc,
        },
        "resources": {
            "upstream_runtime_seconds": upstream_seconds,
            "upstream_peak_rss_kib": max(item["peak_rss_kib"] for item in executions),
            "comparison_seconds": comparison_seconds,
            "runtime_overhead_percent": 100.0 * comparison_seconds / upstream_seconds,
            "checkpoint_max_bytes": max(path.stat().st_size for path in baseline_paths.values()),
            "snapshot_max_bytes": max(snapshot_sizes),
            "comparison_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "executions": executions,
    }


def run_matrix(external_root: Path, output: Path, *, projects: set[str] | None = None, timeout: float = 300.0) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    run_root = output.parent / "runs"
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing run evidence: {run_root}")
    environment = dict(os.environ)
    path_parts = [str(repo_root)]
    site = Path(os.environ.get("TRAINPARITY_GATE4_SITE", ""))
    if str(site):
        path_parts.append(str(site))
    path_parts.append(str(external_root / "ignite"))
    if environment.get("PYTHONPATH"):
        path_parts.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(path_parts)
    selected = [adapter for adapter in ADAPTERS if projects is None or adapter.name in projects]
    records = [
        _project(adapter, root=run_root / adapter.name, checkout=external_root / _checkout_name(adapter), environment=environment, timeout=timeout)
        for adapter in selected
    ]
    adapter_locs = [record["loc"]["adapter_logical"] for record in records]
    report = {
        "schema_version": 1,
        "gate": 4,
        "projects": records,
        "metrics": {
            "project_count": len(records),
            "clean_passed": sum(record["clean"]["outcome"] == "PASS" for record in records),
            "faults_detected": sum(record["fault_result"]["outcome"] == "FAIL" for record in records),
            "adapter_median_logical_loc": statistics.median(adapter_locs),
            "upstream_modified_loc": sum(record["repository"]["upstream_modified_loc"] for record in records),
        },
        "shared_integration_logical_loc": _logical_lines(Path(__file__)) + _logical_lines(Path(__file__).with_name("models.py")),
        "handwritten_comparator_logical_loc": _logical_lines(Path(__file__).with_name("handwritten.py")),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _checkout_name(adapter: ExternalProjectAdapter) -> str:
    return {"pytorch_examples_imagenet": "pytorch-examples", "nanogpt": "nanogpt", "ignite_mnist_engine": "ignite"}[adapter.name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", action="append")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    report = run_matrix(args.external_root, args.output, projects=None if args.project is None else set(args.project), timeout=args.timeout)
    print(json.dumps(report["metrics"], sort_keys=True))
    if report["metrics"]["clean_passed"] != report["metrics"]["project_count"] or report["metrics"]["faults_detected"] != report["metrics"]["project_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
