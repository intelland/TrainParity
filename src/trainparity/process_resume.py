"""Framework-neutral orchestration for command-oriented resume tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

from trainparity.comparison import ExactComparison
from trainparity.importing import CaseImportError, load_process_case
from trainparity.outcomes import Outcome
from trainparity.protocols import ProcessExecutionPlan, ProcessResumeCase
from trainparity.results import ExternalProcessEvidence, ProcessResumeResult
from trainparity.serialization import decode_snapshot
from trainparity.snapshot import Snapshot


@dataclass(frozen=True)
class _ExecutionResult:
    outcome: Outcome
    message: str
    evidence: ExternalProcessEvidence | None = None
    checkpoint: Path | None = None


@dataclass(frozen=True)
class _SnapshotResult:
    outcome: Outcome
    message: str
    snapshot: Snapshot | None = None
    ipc_bytes: int = 0
    elapsed_seconds: float = 0.0
    capture_seconds: float = 0.0
    serialization_seconds: float = 0.0


class ProcessResumeRunner:
    """Run external continuous/resume commands with generic four-state semantics."""

    def __init__(self, *, timeout: float = 300.0, temporary_root: Path | None = None) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self.temporary_root = temporary_root

    def run(
        self,
        case: str,
        *,
        cwd: Path | None = None,
        work_dir: Path | None = None,
        report_path: Path | None = None,
        environment: Mapping[str, str] | None = None,
        staged_checkpoint_hook: Callable[[Path], None] | None = None,
    ) -> ProcessResumeResult:
        """Execute a baseline self-check and fresh-process interrupted candidate."""
        started = time.perf_counter()
        selected_cwd = (cwd or Path.cwd()).resolve()
        environment_updates = dict(environment or {})
        try:
            selected_case = load_process_case(case)
            self._validate_case(selected_case)
            child_environment, propagated_keys = self._child_environment(
                case, environment_updates
            )
        except (CaseImportError, OSError, ValueError) as error:
            result = ProcessResumeResult(
                Outcome.ERROR,
                f"process case setup failed: {type(error).__name__}",
                case,
            )
            return self._finish(result, started, report_path)
        if work_dir is not None:
            work_dir.mkdir(parents=True, exist_ok=True)
            result = self._run(
                case,
                selected_case,
                selected_cwd,
                work_dir.resolve(),
                child_environment,
                propagated_keys,
                staged_checkpoint_hook,
            )
            return self._finish(result, started, report_path)
        if self.temporary_root is not None:
            self.temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="trainparity-process-",
            dir=None if self.temporary_root is None else self.temporary_root,
        ) as temporary:
            result = self._run(
                case,
                selected_case,
                selected_cwd,
                Path(temporary),
                child_environment,
                propagated_keys,
                staged_checkpoint_hook,
            )
            return self._finish(result, started, report_path)

    def _run(
        self,
        case_spec: str,
        case: ProcessResumeCase,
        cwd: Path,
        root: Path,
        environment: dict[str, str],
        propagated_keys: tuple[str, ...],
        staged_checkpoint_hook: Callable[[Path], None] | None,
    ) -> ProcessResumeResult:
        timings: dict[str, float] = {}
        processes: list[ExternalProcessEvidence] = []
        checkpoints: list[Path] = []
        snapshots: dict[str, Snapshot] = {}
        ipc_bytes = 0

        for phase in ("baseline_a", "baseline_b"):
            execution = self._execute(
                case,
                ProcessExecutionPlan(phase, cwd, root / phase, case.total_step),
                environment,
            )
            early = self._execution_problem(
                case_spec, execution, processes, propagated_keys, timings
            )
            if early is not None:
                return early
            assert execution.evidence is not None and execution.checkpoint is not None
            processes.append(execution.evidence)
            checkpoints.append(execution.checkpoint)
            timings[phase] = execution.evidence.elapsed_seconds
            captured = self._capture(
                case_spec,
                execution.checkpoint,
                case.total_step,
                root / "snapshots" / f"{phase}.json",
                cwd,
                environment,
            )
            early = self._snapshot_problem(
                case_spec, captured, processes, propagated_keys, timings
            )
            if early is not None:
                return early
            assert captured.snapshot is not None
            snapshots[phase] = captured.snapshot
            ipc_bytes += captured.ipc_bytes
            timings[f"snapshot_{phase}"] = captured.elapsed_seconds
            timings["snapshot_capture"] = (
                timings.get("snapshot_capture", 0.0) + captured.capture_seconds
            )
            timings["serialization"] = (
                timings.get("serialization", 0.0) + captured.serialization_seconds
            )

        comparison_started = time.perf_counter()
        baseline_differences = ExactComparison().compare_all(
            snapshots["baseline_a"], snapshots["baseline_b"]
        )
        timings["baseline_comparison"] = time.perf_counter() - comparison_started
        if baseline_differences:
            return ProcessResumeResult(
                Outcome.ABSTAIN,
                "baseline self-consistency failed; resume attribution is unsafe",
                case_spec,
                first_divergent_step=case.total_step,
                primary_difference=baseline_differences[0],
                all_differences=baseline_differences,
                processes=tuple(processes),
                propagated_environment_keys=propagated_keys,
                timing_seconds=timings,
                snapshot_ipc_bytes=ipc_bytes,
                checkpoint_max_bytes=self._max_size(checkpoints),
            )

        split = self._execute(
            case,
            ProcessExecutionPlan(
                "candidate_split", cwd, root / "candidate_split", case.split_step
            ),
            environment,
        )
        early = self._execution_problem(case_spec, split, processes, propagated_keys, timings)
        if early is not None:
            return early
        assert split.evidence is not None and split.checkpoint is not None
        processes.append(split.evidence)
        checkpoints.append(split.checkpoint)
        timings["candidate_save_exit"] = split.evidence.elapsed_seconds

        resume_dir = root / "candidate_resume"
        resume_dir.mkdir(parents=True, exist_ok=True)
        staged = case.checkpoint_path(resume_dir)
        staged.parent.mkdir(parents=True, exist_ok=True)
        stage_started = time.perf_counter()
        try:
            shutil.copy2(split.checkpoint, staged)
            if staged_checkpoint_hook is not None:
                staged_checkpoint_hook(staged)
        except (OSError, ValueError, TypeError):
            return ProcessResumeResult(
                Outcome.ERROR,
                "checkpoint staging or staged hook failed",
                case_spec,
                processes=tuple(processes),
                propagated_environment_keys=propagated_keys,
                timing_seconds=timings,
                snapshot_ipc_bytes=ipc_bytes,
                checkpoint_max_bytes=self._max_size(checkpoints),
            )
        timings["checkpoint_staging"] = time.perf_counter() - stage_started

        resumed = self._execute(
            case,
            ProcessExecutionPlan(
                "candidate_resume", cwd, resume_dir, case.total_step, staged
            ),
            environment,
        )
        early = self._execution_problem(case_spec, resumed, processes, propagated_keys, timings)
        if early is not None:
            return early
        assert resumed.evidence is not None and resumed.checkpoint is not None
        processes.append(resumed.evidence)
        checkpoints.append(resumed.checkpoint)
        timings["candidate_new_process_load_resume"] = resumed.evidence.elapsed_seconds
        distinct = split.evidence.pid != resumed.evidence.pid
        if not distinct:
            return ProcessResumeResult(
                Outcome.ERROR,
                "save/exit and load/resume did not prove distinct process IDs",
                case_spec,
                processes=tuple(processes),
                propagated_environment_keys=propagated_keys,
                timing_seconds=timings,
                snapshot_ipc_bytes=ipc_bytes,
                checkpoint_max_bytes=self._max_size(checkpoints),
            )

        captured = self._capture(
            case_spec,
            resumed.checkpoint,
            case.total_step,
            root / "snapshots" / "candidate_resume.json",
            cwd,
            environment,
        )
        early = self._snapshot_problem(case_spec, captured, processes, propagated_keys, timings)
        if early is not None:
            return early
        assert captured.snapshot is not None
        ipc_bytes += captured.ipc_bytes
        timings["snapshot_candidate_resume"] = captured.elapsed_seconds
        timings["snapshot_capture"] += captured.capture_seconds
        timings["serialization"] += captured.serialization_seconds
        comparison_started = time.perf_counter()
        differences = ExactComparison().compare_all(
            snapshots["baseline_a"], captured.snapshot
        )
        timings["candidate_comparison"] = time.perf_counter() - comparison_started
        timings["comparison"] = (
            timings["baseline_comparison"] + timings["candidate_comparison"]
        )
        timings["single_normal_run"] = timings["baseline_a"]
        timings["baseline_self_consistency"] = (
            timings["baseline_a"]
            + timings["baseline_b"]
            + timings["snapshot_baseline_a"]
            + timings["snapshot_baseline_b"]
            + timings["baseline_comparison"]
        )
        timings["candidate_save_exit_new_process_load_resume"] = (
            timings["candidate_save_exit"]
            + timings["checkpoint_staging"]
            + timings["candidate_new_process_load_resume"]
        )
        if differences:
            return ProcessResumeResult(
                Outcome.FAIL,
                f"first observed divergence at step {case.total_step}: {differences[0].path}",
                case_spec,
                first_divergent_step=case.total_step,
                primary_difference=differences[0],
                all_differences=differences,
                processes=tuple(processes),
                fresh_resume_processes_distinct=True,
                propagated_environment_keys=propagated_keys,
                timing_seconds=timings,
                snapshot_ipc_bytes=ipc_bytes,
                checkpoint_max_bytes=self._max_size(checkpoints),
            )
        return ProcessResumeResult(
            Outcome.PASS,
            f"final states are exactly equivalent at step {case.total_step}",
            case_spec,
            processes=tuple(processes),
            fresh_resume_processes_distinct=True,
            propagated_environment_keys=propagated_keys,
            timing_seconds=timings,
            snapshot_ipc_bytes=ipc_bytes,
            checkpoint_max_bytes=self._max_size(checkpoints),
        )

    def _execute(
        self,
        case: ProcessResumeCase,
        plan: ProcessExecutionPlan,
        environment: dict[str, str],
    ) -> _ExecutionResult:
        plan.run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = plan.run_dir / "stdout.log"
        stderr_path = plan.run_dir / "stderr.log"
        try:
            raw_command = case.command(plan)
            command = self._validated_command(raw_command)
            started = time.perf_counter()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=plan.cwd,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                )
                try:
                    returncode = process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    return _ExecutionResult(
                        Outcome.ERROR,
                        f"{plan.phase} child exceeded timeout",
                    )
            elapsed = time.perf_counter() - started
        except (OSError, TypeError, ValueError):
            return _ExecutionResult(Outcome.ERROR, f"{plan.phase} child launch failed")
        evidence = ExternalProcessEvidence(plan.phase, process.pid, elapsed, returncode)
        if returncode != 0:
            return _ExecutionResult(
                Outcome.ERROR,
                f"{plan.phase} child exited with status {returncode}",
                evidence,
            )
        checkpoint = case.checkpoint_path(plan.run_dir)
        if not checkpoint.is_file():
            return _ExecutionResult(
                Outcome.ERROR,
                f"{plan.phase} did not create its declared checkpoint",
                evidence,
            )
        return _ExecutionResult(Outcome.PASS, "child completed", evidence, checkpoint)

    def _capture(
        self,
        case: str,
        checkpoint: Path,
        step: int,
        result_path: Path,
        cwd: Path,
        environment: dict[str, str],
    ) -> _SnapshotResult:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "trainparity.process_worker",
            "--case",
            case,
            "--checkpoint",
            str(checkpoint),
            "--step",
            str(step),
            "--result",
            str(result_path),
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return _SnapshotResult(Outcome.ERROR, "snapshot worker launch or timeout failed")
        elapsed = time.perf_counter() - started
        if completed.returncode != 0 or not result_path.is_file():
            return _SnapshotResult(Outcome.ERROR, "snapshot worker did not publish a result")
        try:
            payload: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
            outcome = Outcome(payload["outcome"])
            if outcome is not Outcome.PASS:
                return _SnapshotResult(outcome, str(payload["message"]), elapsed_seconds=elapsed)
            snapshot = decode_snapshot(payload["snapshot"])
            worker_timings = payload["timing_seconds"]
            return _SnapshotResult(
                Outcome.PASS,
                "snapshot captured",
                snapshot,
                result_path.stat().st_size,
                elapsed,
                float(worker_timings["capture"]),
                float(worker_timings["serialization"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return _SnapshotResult(Outcome.ERROR, "snapshot worker result was corrupt")

    @staticmethod
    def _validate_case(case: ProcessResumeCase) -> None:
        if not case.name or not 0 < case.split_step < case.total_step:
            raise ValueError("process case name/steps are invalid")

    @staticmethod
    def _validated_command(command: Sequence[str]) -> list[str]:
        if not command or any(not isinstance(value, str) or "\x00" in value for value in command):
            raise ValueError("case command must be a non-empty string sequence")
        return list(command)

    @staticmethod
    def _child_environment(
        case: str, updates: Mapping[str, str]
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        environment = dict(os.environ)
        for key, value in updates.items():
            if not key or "=" in key or "\x00" in key or "\x00" in value:
                raise ValueError("child environment contains an invalid key or value")
            environment[key] = value
        module_name = case.partition(":")[0]
        module = import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            parent = str(Path(module_file).resolve().parent)
            existing = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = parent if not existing else os.pathsep.join((parent, existing))
        return environment, tuple(sorted(updates))

    @staticmethod
    def _execution_problem(
        case: str,
        execution: _ExecutionResult,
        processes: list[ExternalProcessEvidence],
        propagated_keys: tuple[str, ...],
        timings: dict[str, float],
    ) -> ProcessResumeResult | None:
        if execution.outcome is Outcome.PASS:
            return None
        if execution.evidence is not None:
            processes.append(execution.evidence)
            timings[execution.evidence.phase] = execution.evidence.elapsed_seconds
        return ProcessResumeResult(
            Outcome.ERROR,
            execution.message,
            case,
            processes=tuple(processes),
            propagated_environment_keys=propagated_keys,
            timing_seconds=timings,
        )

    @staticmethod
    def _snapshot_problem(
        case: str,
        snapshot: _SnapshotResult,
        processes: list[ExternalProcessEvidence],
        propagated_keys: tuple[str, ...],
        timings: dict[str, float],
    ) -> ProcessResumeResult | None:
        if snapshot.outcome is Outcome.PASS:
            return None
        outcome = Outcome.ABSTAIN if snapshot.outcome is Outcome.ABSTAIN else Outcome.ERROR
        return ProcessResumeResult(
            outcome,
            snapshot.message,
            case,
            processes=tuple(processes),
            propagated_environment_keys=propagated_keys,
            timing_seconds=timings,
        )

    @staticmethod
    def _max_size(paths: Sequence[Path]) -> int:
        return max((path.stat().st_size for path in paths if path.is_file()), default=0)

    @staticmethod
    def _finish(
        result: ProcessResumeResult,
        started: float,
        report_path: Path | None,
    ) -> ProcessResumeResult:
        timings = dict(result.timing_seconds or {})
        timings["total_wall"] = time.perf_counter() - started
        normal = timings.get("single_normal_run")
        if normal is not None and normal > 0:
            timings["end_to_end_multiplier"] = timings["total_wall"] / normal
        completed = replace(result, timing_seconds=timings)
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_suffix(report_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(completed.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(report_path)
        return completed
