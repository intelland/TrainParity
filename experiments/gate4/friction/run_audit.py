"""Rework Gate 4 with fresh-clone demonstrations and honest end-to-end costs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECTS = (
    {
        "name": "pytorch_examples_imagenet",
        "checkout": "pytorch-examples",
        "repository": "https://github.com/pytorch/examples.git",
        "commit": "acc295dc7b90714f1bf47f06004fc19a7fe235c4",
        "license": "BSD-3-Clause",
        "structure": "conventional image classifier",
        "fault": "scheduler_last_epoch_off_by_one",
        "fault_class": "control-state",
    },
    {
        "name": "nanogpt",
        "checkout": "nanogpt",
        "repository": "https://github.com/karpathy/nanoGPT.git",
        "commit": "3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
        "license": "MIT",
        "structure": "small language model",
        "fault": "resume_iteration_off_by_one",
        "fault_class": "trajectory-affecting",
    },
    {
        "name": "ignite_mnist_engine",
        "checkout": "ignite",
        "repository": "https://github.com/pytorch/ignite.git",
        "commit": "e08ff9257ed18d8d805304e32ba85a44553195fc",
        "license": "BSD-3-Clause",
        "structure": "trainer engine with extra resumable state",
        "fault": "scheduler_last_epoch_off_by_one",
        "fault_class": "control-state",
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
    timeout: float = 600.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command!r}\n{completed.stderr[-4000:]}"
        )
    return {
        "command": command,
        "cwd": str(cwd),
        "elapsed_seconds": elapsed,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def _evidence_hashes(repo_root: Path) -> dict[str, str]:
    paths = sorted((repo_root / "artifacts" / "gate_reports").glob("gate_[0-4].*"))
    for gate in range(5):
        paths.extend(sorted((repo_root / "experiments" / f"gate{gate}" / "recorded").glob("*")))
    return {
        str(path.relative_to(repo_root)): _sha256(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


def _fresh_project(
    metadata: dict[str, str],
    *,
    repo_root: Path,
    output_root: Path,
    wheel: Path,
    gate4_site: Path,
) -> dict[str, Any]:
    name = metadata["name"]
    clone = output_root / "clones" / metadata["checkout"]
    project_root = output_root / "projects" / name
    user_directory = clone / ".trainparity_user"
    install_directory = clone / ".trainparity_site"
    commands: list[dict[str, Any]] = []
    commands.append(
        _run(["git", "clone", "--no-checkout", metadata["repository"], str(clone)], cwd=output_root)
    )
    commands.append(
        _run(["git", "checkout", "--detach", metadata["commit"]], cwd=clone)
    )
    observed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=clone, text=True
    ).strip()
    if observed_commit != metadata["commit"]:
        raise RuntimeError(f"{name}: expected {metadata['commit']}, observed {observed_commit}")

    user_directory.mkdir(parents=True)
    common_source = repo_root / "experiments/gate4/friction/user_files/trainparity_clean_resume.py"
    project_source = repo_root / "experiments/gate4/friction/user_files" / name
    source_files = {
        "trainparity_adapter.py": project_source / "trainparity_adapter.py",
        "trainparity_project_glue.py": project_source / "trainparity_project_glue.py",
        "trainparity_clean_resume.py": common_source,
    }
    for filename, source in source_files.items():
        shutil.copy2(source, user_directory / filename)

    install_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(install_directory),
        str(wheel),
    ]
    commands.append(_run(install_command, cwd=clone))
    environment = dict(os.environ)
    python_paths = [str(install_directory), str(gate4_site)]
    if name == "ignite_mnist_engine":
        python_paths.append(str(clone))
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    environment.setdefault("TRAINPARITY_GATE4_DEVICE", "cuda")
    import_check = [
        sys.executable,
        "-c",
        "import pathlib,trainparity; print(pathlib.Path(trainparity.__file__).resolve())",
    ]
    import_record = _run(import_check, cwd=clone, environment=environment)
    commands.append(import_record)
    if str(install_directory.resolve()) not in import_record["stdout"]:
        raise RuntimeError(f"{name}: TrainParity did not import from the wheel target")

    output = project_root / "clean_resume.json"
    clean_command = [
        sys.executable,
        str(user_directory / "trainparity_clean_resume.py"),
        "--checkout",
        str(clone),
        "--workspace",
        str(project_root / "workspace"),
        "--output",
        str(output),
    ]
    commands.append(_run(clean_command, cwd=clone, environment=environment, timeout=900.0))
    clean = json.loads(output.read_text(encoding="utf-8"))
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--numstat", metadata["commit"], "--"], cwd=clone, text=True
    ).strip()
    modified_loc = 0
    for line in tracked_diff.splitlines():
        added, deleted, _ = line.split("\t", 2)
        modified_loc += int(added) + int(deleted)

    adapter_loc = _logical_lines(source_files["trainparity_adapter.py"])
    supporting_files = [
        source_files["trainparity_project_glue.py"],
        source_files["trainparity_clean_resume.py"],
    ]
    supporting_loc = sum(_logical_lines(path) for path in supporting_files)
    user_files = [
        {
            "path": f".trainparity_user/{filename}",
            "role": "adapter" if filename == "trainparity_adapter.py" else "supporting_glue",
            "logical_loc": _logical_lines(source),
            "sha256": _sha256(source),
        }
        for filename, source in source_files.items()
    ]
    license_path = clone / "LICENSE"
    return {
        "name": name,
        "structure": metadata["structure"],
        "repository": metadata["repository"],
        "commit": observed_commit,
        "license": metadata["license"],
        "license_file": str(license_path),
        "license_sha256": _sha256(license_path),
        "fresh_clone": True,
        "wheel_installed": str(wheel.name),
        "trainparity_import_path": import_record["stdout"].strip(),
        "user_required_files": user_files,
        "loc": {
            "user_required_adapter": adapter_loc,
            "user_required_supporting_glue": supporting_loc,
            "total_user_required": adapter_loc + supporting_loc,
            "upstream_modified": modified_loc,
        },
        "clean_resume": clean,
        "exact_commands": [record["command"] for record in commands],
        "command_records": commands,
        "environment": {
            "PYTHONPATH": environment["PYTHONPATH"],
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"],
            "TRAINPARITY_GATE4_DEVICE": environment["TRAINPARITY_GATE4_DEVICE"],
        },
        "clone_path": str(clone),
        "user_directory": str(user_directory),
    }


def _fault_evidence(recorded: dict[str, Any]) -> list[dict[str, Any]]:
    classifications = {item["name"]: item for item in PROJECTS}
    evidence = []
    for project in recorded["projects"]:
        differences = project["fault_result"]["all_differences"]
        model_prefix = "model" if project["name"] == "nanogpt" else "state_dict"
        downstream = any(item["path"].startswith(model_prefix) for item in differences)
        evidence.append(
            {
                "project": project["name"],
                "fault": project["fault"],
                "classification": classifications[project["name"]]["fault_class"],
                "first_observed_divergence": project["fault_result"]["primary_difference"],
                "downstream_parameter_divergence_observed": downstream,
                "difference_count_at_first_divergent_step": len(differences),
            }
        )
    return evidence


def _loc_inventory(repo_root: Path, projects: list[dict[str, Any]]) -> dict[str, Any]:
    reusable_files = [
        repo_root / "src/trainparity/state.py",
        repo_root / "src/trainparity/snapshot.py",
        repo_root / "src/trainparity/serialization.py",
        repo_root / "src/trainparity/comparison.py",
    ]
    reusable = [
        {"path": str(path.relative_to(repo_root)), "logical_loc": _logical_lines(path)}
        for path in reusable_files
    ]
    project_gate4_files: dict[str, list[Path]] = {
        "pytorch_examples_imagenet": [
            repo_root / "experiments/gate4/adapters/pytorch_examples_imagenet.py",
            repo_root / "experiments/gate4/drivers/imagenet.py",
        ],
        "nanogpt": [
            repo_root / "experiments/gate4/adapters/nanogpt.py",
            repo_root / "experiments/gate4/drivers/nanogpt.py",
        ],
        "ignite_mnist_engine": [
            repo_root / "experiments/gate4/adapters/ignite_mnist.py",
            repo_root / "experiments/gate4/drivers/ignite_mnist.py",
        ],
    }
    project_specific = {
        name: {
            "files": [
                {"path": str(path.relative_to(repo_root)), "logical_loc": _logical_lines(path)}
                for path in paths
            ],
            "logical_loc": sum(_logical_lines(path) for path in paths),
        }
        for name, paths in project_gate4_files.items()
    }
    excluded = {path.resolve() for paths in project_gate4_files.values() for path in paths}
    shared_paths = [
        path
        for path in (repo_root / "experiments/gate4").rglob("*.py")
        if "user_files" not in path.parts and path.resolve() not in excluded
    ]
    for relative in (
        "scripts/verify_gate4_friction_audit.py",
        "scripts/slurm_gate4_friction_audit.sbatch",
        "tests/test_gate4_friction_audit.py",
    ):
        path = repo_root / relative
        if path.exists():
            shared_paths.append(path)
    shared = [
        {"path": str(path.relative_to(repo_root)), "logical_loc": _logical_lines(path)}
        for path in sorted(set(shared_paths))
    ]
    return {
        "reusable_trainparity_library": {
            "scope": "complete source modules imported by the user integration; not per-project new LOC",
            "files": reusable,
            "logical_loc": sum(item["logical_loc"] for item in reusable),
        },
        "gate4_only_project_specific": project_specific,
        "gate4_only_shared": {
            "scope": "benchmark, fault injection, measurement, reporting, and verification; not user integration",
            "files": shared,
            "logical_loc": sum(item["logical_loc"] for item in shared),
        },
        "per_project_categories": {
            project["name"]: {
                "a_user_adapter_loc": project["loc"]["user_required_adapter"],
                "b_user_supporting_glue_loc": project["loc"]["user_required_supporting_glue"],
                "c_reusable_library_loc_shared_not_new": sum(
                    item["logical_loc"] for item in reusable
                ),
                "d_gate4_only_project_specific_loc": project_specific[project["name"]][
                    "logical_loc"
                ],
                "d_gate4_only_shared_loc_not_allocated": sum(
                    item["logical_loc"] for item in shared
                ),
            }
            for project in projects
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Gate 4 Friction Audit",
        "",
        f"**Audit outcome:** {report['outcome']}",
        "",
        "This is a Gate 4 rework audit only. Gate 5 was not started.",
        "",
        "## Fresh-clone user cost",
        "",
        "| Project | Adapter LOC | Supporting glue LOC | Total user LOC | Upstream LOC | Clean | E2E multiplier |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for project in report["projects"]:
        loc = project["loc"]
        timing = project["clean_resume"]["timing_seconds"]
        lines.append(
            f"| {project['name']} | {loc['user_required_adapter']} | "
            f"{loc['user_required_supporting_glue']} | {loc['total_user_required']} | "
            f"{loc['upstream_modified']} | {project['clean_resume']['outcome']} | "
            f"{timing['end_to_end_multiplier']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "Every row was reproduced from a new exact-commit clone, a no-dependencies wheel install, "
            "and only the three listed user source files. Generated data, checkpoints, logs, and the "
            "isolated wheel target are runtime artifacts, not hidden integration source.",
            "",
            "The supporting glue exceeds 50 LOC in every case because the current production API does "
            "not orchestrate command-oriented external repositories. The audit counts process launch, "
            "baseline repetition, checkpoint staging, snapshot serialization, and reporting rather than "
            "hiding them in experiment helpers.",
            "",
            "## End-to-end measurements",
            "",
            "| Project | Normal | Baseline self-check | Save/exit + new-process/load/resume | Snapshot | Serialize | Compare | Total | Peak RSS KiB | Artifacts bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for project in report["projects"]:
        clean = project["clean_resume"]
        timing = clean["timing_seconds"]
        lines.append(
            f"| {project['name']} | {timing['single_normal_run']:.4f}s | "
            f"{timing['baseline_self_consistency']:.4f}s | "
            f"{timing['candidate_save_exit_new_process_load_resume']:.4f}s | "
            f"{timing['snapshot_capture']:.4f}s | {timing['serialization']:.4f}s | "
            f"{timing['comparison']:.6f}s | {timing['total_wall']:.4f}s | "
            f"{clean['peak_rss_kib']} | {clean['total_artifact_size_bytes']} |"
        )
    lines.extend(
        [
            "",
            "The earlier comparator-only timing is preserved in the JSON report and is explicitly not "
            "the total TrainParity overhead. Phase aggregates overlap where labeled: baseline "
            "self-consistency includes both normal runs, while snapshot, serialization, and comparison "
            "are also shown separately.",
            "",
            "## Hand-written controls",
            "",
            f"The existing {report['baselines']['weak']['logical_loc']}-line final-model-only comparator "
            "is retained as a **weak baseline**. It omits optimizer, scheduler, RNG, process orchestration, "
            "and path-level diagnostics.",
            "",
            f"The closer hand-written Ignite test is project-specific and has "
            f"{report['baselines']['closer']['logical_loc']} logical lines (plus the explicitly reported "
            "user glue dependency). It checks model, optimizer, scheduler, and torch CPU RNG across a "
            "fresh-process resume. Its fault diagnostic is: "
            f"`{report['baselines']['closer']['result']['diagnostic']}`.",
            "",
            "## Fault classification",
            "",
            "| Project | Classification | First observed divergence | Downstream parameters diverged |",
            "|---|---|---|---|",
        ]
    )
    for fault in report["faults"]:
        path = fault["first_observed_divergence"]["path"]
        lines.append(
            f"| {fault['project']} | {fault['classification']} | `{path}` | "
            f"{str(fault['downstream_parameter_divergence_observed']).lower()} |"
        )
    lines.extend(
        [
            "",
            "These are classifications and first observed divergences, not root-cause claims. No "
            "reporting-only fault was injected in the accepted three-project suite.",
            "",
            "## Preservation and scope",
            "",
            f"- Accepted Gate 0-4 evidence files checked: {len(report['preservation']['accepted_evidence_sha256'])}",
            f"- User document SHA-256 (local pre-run observation): `{report['preservation']['user_document_sha256']}`",
            "- Production API changes: none",
            "- New framework adapters: none",
            "- Snapshot backend optimization: none",
            "- New projects: none",
            "- Gate 5 work: none",
            "",
            "## Exact commands",
            "",
            "See `projects[].exact_commands` in the JSON report for every clone, checkout, install, "
            "import, and clean-resume command. Verification commands are also recorded there.",
            "",
            "## Recommendation",
            "",
            report["recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    output_root: Path,
    report_json: Path,
    report_markdown: Path,
    user_document_sha256: str,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite audit evidence: {output_root}")
    output_root.mkdir(parents=True)
    evidence_before = _evidence_hashes(repo_root)
    wheels = sorted((repo_root / "dist").glob("trainparity-*.whl"))
    if not wheels:
        raise RuntimeError("no TrainParity wheel found")
    wheel = wheels[-1]
    site = Path(os.environ["TRAINPARITY_GATE4_SITE"]).resolve()
    projects = [
        _fresh_project(
            dict(metadata),
            repo_root=repo_root,
            output_root=output_root,
            wheel=wheel.resolve(),
            gate4_site=site,
        )
        for metadata in PROJECTS
    ]

    ignite = next(project for project in projects if project["name"] == "ignite_mnist_engine")
    strong_output = output_root / "handwritten_closer.json"
    strong_command = [
        sys.executable,
        str(repo_root / "experiments/gate4/friction/handwritten_fresh_resume.py"),
        "--checkout",
        ignite["clone_path"],
        "--workspace",
        str(output_root / "projects/ignite_mnist_engine/workspace"),
        "--user-files",
        ignite["user_directory"],
        "--output",
        str(strong_output),
    ]
    strong_environment = dict(os.environ)
    strong_environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(ignite["clone_path"]) / ".trainparity_site"),
            str(site),
            ignite["clone_path"],
        ]
    )
    strong_environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    strong_environment["TRAINPARITY_GATE4_DEVICE"] = os.environ.get(
        "TRAINPARITY_GATE4_DEVICE", "cuda"
    )
    strong_record = _run(
        strong_command,
        cwd=Path(ignite["clone_path"]),
        environment=strong_environment,
        timeout=600.0,
    )
    strong_result = json.loads(strong_output.read_text(encoding="utf-8"))
    recorded = json.loads(
        (repo_root / "experiments/gate4/recorded/matrix.json").read_text(encoding="utf-8")
    )
    evidence_after = _evidence_hashes(repo_root)
    if evidence_before != evidence_after:
        raise RuntimeError("accepted Gate 0-4 evidence changed during friction audit")
    loc_inventory = _loc_inventory(repo_root, projects)
    adapter_locs = [project["loc"]["user_required_adapter"] for project in projects]
    clean_count = sum(project["clean_resume"]["outcome"] == "PASS" for project in projects)
    weak = {
        "label": "weak final-model-only baseline",
        "logical_loc": _logical_lines(repo_root / "experiments/gate4/handwritten.py"),
        "limitations": [
            "same-process value comparator only",
            "model final state only",
            "no optimizer, scheduler, or RNG",
            "no first-observed path beyond equal/not-equal",
        ],
        "accepted_gate4_results": {
            project["name"]: project["handwritten"] for project in recorded["projects"]
        },
    }
    closer_path = repo_root / "experiments/gate4/friction/handwritten_fresh_resume.py"
    closer = {
        "label": "functionally closer hand-written fresh-process resume test",
        "logical_loc": _logical_lines(closer_path),
        "project_specific": True,
        "project": "ignite_mnist_engine",
        "dependent_user_integration_loc": next(
            project["loc"]["total_user_required"]
            for project in projects
            if project["name"] == "ignite_mnist_engine"
        ),
        "result": strong_result,
        "exact_command": strong_command,
        "command_record": strong_record,
    }
    all_pass = (
        clean_count == 3
        and all(project["loc"]["upstream_modified"] == 0 for project in projects)
        and strong_result["clean"]["outcome"] == "PASS"
        and strong_result["fault"]["outcome"] == "FAIL"
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "gate": 4,
        "audit": "friction_rework",
        "outcome": "PASS" if all_pass else "FAIL",
        "recommendation": (
            "REWORK: the technical GO evidence remains positive, but the honest user-required glue is "
            "well above 50 LOC because command-oriented orchestration is not in the production API. "
            "Do not begin Gate 5; human review should decide whether this friction is acceptable."
        ),
        "metrics": {
            "project_count": len(projects),
            "fresh_clone_clean_passed": clean_count,
            "adapter_median_logical_loc": statistics.median(adapter_locs),
            "upstream_modified_loc": sum(project["loc"]["upstream_modified"] for project in projects),
        },
        "projects": projects,
        "loc_inventory": loc_inventory,
        "comparator_only_timing_preserved_not_total_overhead": {
            project["name"]: {
                "comparison_seconds": project["resources"]["comparison_seconds"],
                "formerly_reported_runtime_overhead_percent": project["resources"][
                    "runtime_overhead_percent"
                ],
            }
            for project in recorded["projects"]
        },
        "baselines": {"weak": weak, "closer": closer},
        "faults": _fault_evidence(recorded),
        "preservation": {
            "accepted_evidence_sha256": evidence_after,
            "accepted_evidence_unchanged_during_audit": True,
            "user_document_sha256": user_document_sha256,
            "user_document_note": "hash supplied from the untouched local dirty working copy",
        },
        "scope_guards": {
            "production_api_changed": False,
            "framework_adapters_added": False,
            "snapshot_backend_optimized": False,
            "new_projects_added": False,
            "gate5_started": False,
        },
        "verification_commands": [
            "make lint",
            "make typecheck",
            "make test",
            "python scripts/verify_gate.py 4",
            "python scripts/verify_gate4_friction_audit.py",
            "git diff --check",
        ],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-markdown", type=Path, required=True)
    parser.add_argument("--user-document-sha256", required=True)
    args = parser.parse_args()
    report = run_audit(
        args.output_root.resolve(),
        args.report_json.resolve(),
        args.report_markdown.resolve(),
        args.user_document_sha256,
    )
    print(json.dumps(report["metrics"], sort_keys=True))
    if report["outcome"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
