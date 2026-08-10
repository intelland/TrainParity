"""Run the Gate 4B production surface against three fresh pinned clones."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECTS = (
    {
        "name": "pytorch_examples_imagenet",
        "checkout": "pytorch-examples",
        "repository": "https://github.com/pytorch/examples.git",
        "commit": "acc295dc7b90714f1bf47f06004fc19a7fe235c4",
        "license": "BSD-3-Clause",
        "structure": "conventional image classifier",
        "checkpoint": {
            "save": "imagenet/main.py save_checkpoint",
            "load": "imagenet/main.py --resume torch.load and load_state_dict",
        },
    },
    {
        "name": "nanogpt",
        "checkout": "nanogpt",
        "repository": "https://github.com/karpathy/nanoGPT.git",
        "commit": "3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
        "license": "MIT",
        "structure": "small language model",
        "checkpoint": {
            "save": "train.py torch.save checkpoint",
            "load": "train.py init_from=resume torch.load",
        },
    },
    {
        "name": "ignite_mnist_engine",
        "checkout": "ignite",
        "repository": "https://github.com/pytorch/ignite.git",
        "commit": "e08ff9257ed18d8d805304e32ba85a44553195fc",
        "license": "BSD-3-Clause",
        "structure": "trainer engine with extra resumable state",
        "checkpoint": {
            "save": "mnist_save_resume_engine.py Checkpoint and DiskSaver",
            "load": "torch.load and Checkpoint.load_objects",
        },
    },
)


def _logical_lines(path: Path) -> int:
    return sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    timeout: float = 900.0,
    metrics_dir: Path | None = None,
) -> dict[str, Any]:
    actual = command
    time_path = stdout_path = stderr_path = None
    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        time_path = metrics_dir / "time.txt"
        stdout_path = metrics_dir / "stdout.log"
        stderr_path = metrics_dir / "stderr.log"
        actual = ["/usr/bin/time", "-v", "-o", str(time_path), *command]
    started = time.perf_counter()
    if stdout_path is None or stderr_path is None:
        completed = subprocess.run(
            actual,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout, stderr = completed.stdout, completed.stderr
    else:
        with stdout_path.open("w", encoding="utf-8") as stdout_stream, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_stream:
            completed = subprocess.run(
                actual,
                cwd=cwd,
                env=environment,
                text=True,
                stdout=stdout_stream,
                stderr=stderr_stream,
                timeout=timeout,
                check=False,
            )
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {command!r}\n{stderr[-4000:]}")
    peak_rss_kib = 0
    if time_path is not None:
        for line in time_path.read_text(encoding="utf-8").splitlines():
            if "Maximum resident set size" in line:
                peak_rss_kib = int(line.rsplit(":", 1)[1].strip())
    return {
        "command": command,
        "cwd": str(cwd),
        "elapsed_seconds": elapsed,
        "peak_rss_kib": peak_rss_kib,
        "stdout_tail": stdout[-1000:],
        "stderr_tail": stderr[-1000:],
    }


def _prepare(name: str, checkout: Path) -> None:
    if name == "pytorch_examples_imagenet":
        for split in ("train", "val"):
            directory = checkout / ".trainparity_data" / split / "only"
            directory.mkdir(parents=True, exist_ok=True)
            pixels = bytes((128, 128, 128)) * (256 * 256)
            (directory / "0.ppm").write_bytes(b"P6\n256 256\n255\n" + pixels)
    elif name == "nanogpt":
        directory = checkout / "data/tiny_gate4b"
        directory.mkdir(parents=True, exist_ok=True)
        tokens = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
        tokens.tofile(directory / "train.bin")
        tokens.tofile(directory / "val.bin")
        with (directory / "meta.pkl").open("wb") as stream:
            pickle.dump({"vocab_size": 16}, stream)


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _modified_loc(checkout: Path, commit: str) -> int:
    output = subprocess.check_output(
        ["git", "diff", "--numstat", commit, "--"], cwd=checkout, text=True
    ).strip()
    total = 0
    for line in output.splitlines():
        added, deleted, _ = line.split("\t", 2)
        total += int(added) + int(deleted)
    return total


def _project(
    metadata: dict[str, Any],
    *,
    repo_root: Path,
    output_root: Path,
    wheel: Path,
    site: Path,
) -> dict[str, Any]:
    name = str(metadata["name"])
    checkout = output_root / "clones" / str(metadata["checkout"])
    user_dir = checkout / ".trainparity_user"
    install_dir = checkout / ".trainparity_site"
    project_root = output_root / "projects" / name
    records = [
        _run(
            ["git", "clone", "--no-checkout", str(metadata["repository"]), str(checkout)],
            cwd=output_root,
        )
    ]
    records.append(_run(["git", "checkout", "--detach", str(metadata["commit"])], cwd=checkout))
    observed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if observed_commit != metadata["commit"]:
        raise RuntimeError(f"{name}: fresh clone commit mismatch")

    user_dir.mkdir(parents=True)
    adapter_source = repo_root / "experiments/gate4b/user_files" / name / "trainparity_adapter.py"
    invocation_source = repo_root / "experiments/gate4b/user_files/test_resume.py"
    shutil.copy2(adapter_source, user_dir / "trainparity_adapter.py")
    shutil.copy2(invocation_source, user_dir / "test_resume.py")
    records.append(
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(install_dir),
                str(wheel),
            ],
            cwd=checkout,
        )
    )
    _prepare(name, checkout)
    environment = dict(os.environ)
    paths = [str(install_dir), str(user_dir), str(site)]
    if name == "ignite_mnist_engine":
        paths.append(str(checkout))
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["TRAINPARITY_GATE4_DEVICE"] = os.environ.get("TRAINPARITY_GATE4_DEVICE", "cuda")
    environment["TRAINPARITY_GATE4B_IGNITE_DRIVER"] = str(
        repo_root / "experiments/gate4b/ignite_driver.py"
    )
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    import_record = _run(
        [
            sys.executable,
            "-c",
            "import pathlib,trainparity; print(pathlib.Path(trainparity.__file__).resolve())",
        ],
        cwd=checkout,
        environment=environment,
    )
    records.append(import_record)
    if str(install_dir.resolve()) not in import_record["stdout_tail"]:
        raise RuntimeError(f"{name}: TrainParity did not import from the wheel target")

    clean_root = project_root / "clean"
    clean_report_path = clean_root / "report.json"
    clean_record = _run(
        [sys.executable, str(user_dir / "test_resume.py"), str(clean_report_path)],
        cwd=checkout,
        environment=environment,
        metrics_dir=clean_root,
    )
    clean = json.loads(clean_report_path.read_text(encoding="utf-8"))
    clean_artifact_size = _tree_size(clean_root)
    fault_root = project_root / "fault"
    fault_report_path = fault_root / "report.json"
    fault_record = _run(
        [
            sys.executable,
            str(repo_root / "experiments/gate4b/run_case.py"),
            "--project",
            name,
            "--report",
            str(fault_report_path),
            "--temporary-root",
            str(fault_root / "temporary"),
        ],
        cwd=checkout,
        environment=environment,
        metrics_dir=fault_root,
    )
    fault = json.loads(fault_report_path.read_text(encoding="utf-8"))
    adapter_loc = _logical_lines(adapter_source)
    support_loc = _logical_lines(invocation_source)
    checkpoint_max = int(clean["checkpoint_max_bytes"])
    artifact_limit = 5 * checkpoint_max if checkpoint_max >= 1_000_000 else 5_000_000
    total_wall = float(clean["timing_seconds"]["total_wall"])
    normal_wall = float(clean["timing_seconds"]["single_normal_run"])
    return {
        "name": name,
        "structure": metadata["structure"],
        "repository": {
            "url": metadata["repository"],
            "commit": observed_commit,
            "license": metadata["license"],
            "license_sha256": _sha256(checkout / "LICENSE"),
            "upstream_modified_loc": _modified_loc(checkout, observed_commit),
        },
        "checkpoint_implementation": metadata["checkpoint"],
        "user_required_files": [
            {
                "path": ".trainparity_user/trainparity_adapter.py",
                "role": "adapter",
                "logical_loc": adapter_loc,
                "sha256": _sha256(adapter_source),
            },
            {
                "path": ".trainparity_user/test_resume.py",
                "role": "supporting_glue",
                "logical_loc": support_loc,
                "sha256": _sha256(invocation_source),
            },
        ],
        "loc": {
            "adapter_logical": adapter_loc,
            "supporting_glue_logical": support_loc,
            "total_user_logical": adapter_loc + support_loc,
            "upstream_modified": _modified_loc(checkout, observed_commit),
        },
        "clean": clean,
        "fault": fault,
        "resources": {
            "single_normal_run_seconds": normal_wall,
            "total_trainparity_wall_seconds": total_wall,
            "end_to_end_multiplier": total_wall / normal_wall,
            "outer_wall_seconds": clean_record["elapsed_seconds"],
            "peak_rss_kib": clean_record["peak_rss_kib"],
            "checkpoint_max_bytes": checkpoint_max,
            "snapshot_ipc_bytes": clean["snapshot_ipc_bytes"],
            "total_persisted_artifact_bytes": clean_artifact_size,
            "artifact_limit_bytes": artifact_limit,
            "wall_threshold_passed": total_wall <= 6 * normal_wall,
            "artifact_threshold_passed": clean_artifact_size <= artifact_limit,
        },
        "fresh_clone": True,
        "wheel": wheel.name,
        "trainparity_import_path": import_record["stdout_tail"].strip(),
        "environment_propagated_keys": clean["propagated_environment_keys"],
        "exact_commands": [record["command"] for record in [*records, clean_record, fault_record]],
    }


def run_matrix(
    output_root: Path,
    output: Path,
    profile_pre: Path,
    profile_post: Path,
    selected_projects: tuple[str, ...] = (),
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Gate 4B output: {output_root}")
    output_root.mkdir(parents=True)
    wheels = sorted((repo_root / "dist").glob("trainparity-*.whl"))
    if not wheels or "dev5" not in wheels[-1].name:
        raise RuntimeError("Gate 4B requires the dev5 wheel")
    site = Path(os.environ["TRAINPARITY_GATE4_SITE"]).resolve()
    metadata_items = [
        metadata
        for metadata in PROJECTS
        if not selected_projects or metadata["name"] in selected_projects
    ]
    projects = [
        _project(dict(metadata), repo_root=repo_root, output_root=output_root, wheel=wheels[-1], site=site)
        for metadata in metadata_items
    ]
    totals = [project["loc"]["total_user_logical"] for project in projects]
    report = {
        "schema_version": 1,
        "gate": "4B",
        "projects": projects,
        "metrics": {
            "project_count": len(projects),
            "clean_passed": sum(project["clean"]["outcome"] == "PASS" for project in projects),
            "faults_detected": sum(project["fault"]["outcome"] == "FAIL" for project in projects),
            "median_total_user_logical_loc": statistics.median(totals),
            "max_adapter_logical_loc": max(project["loc"]["adapter_logical"] for project in projects),
            "max_supporting_glue_logical_loc": max(project["loc"]["supporting_glue_logical"] for project in projects),
            "max_total_user_logical_loc": max(totals),
            "upstream_modified_loc": sum(project["loc"]["upstream_modified"] for project in projects),
        },
        "snapshot_profile": {
            "pre": json.loads(profile_pre.read_text(encoding="utf-8")),
            "post": json.loads(profile_post.read_text(encoding="utf-8")),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": os.environ.get("TRAINPARITY_GATE4_DEVICE", "cpu"),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-pre", type=Path, required=True)
    parser.add_argument("--profile-post", type=Path, required=True)
    parser.add_argument(
        "--project",
        action="append",
        choices=[str(project["name"]) for project in PROJECTS],
        default=[],
    )
    args = parser.parse_args()
    report = run_matrix(
        args.output_root.resolve(),
        args.output.resolve(),
        args.profile_pre.resolve(),
        args.profile_post.resolve(),
        tuple(args.project),
    )
    print(json.dumps(report["metrics"], sort_keys=True))
    metrics = report["metrics"]
    if (
        metrics["clean_passed"] != metrics["project_count"]
        or metrics["faults_detected"] != metrics["project_count"]
        or metrics["max_adapter_logical_loc"] > 30
        or metrics["max_supporting_glue_logical_loc"] > 20
        or metrics["max_total_user_logical_loc"] > 50
        or metrics["median_total_user_logical_loc"] > 40
        or metrics["upstream_modified_loc"] != 0
        or any(
            not project["resources"]["wall_threshold_passed"]
            or not project["resources"]["artifact_threshold_passed"]
            for project in report["projects"]
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
