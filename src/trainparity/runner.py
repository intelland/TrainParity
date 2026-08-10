"""Orchestrate continuous and real-process resume trajectories."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainparity.comparison import ExactComparison
from trainparity.outcomes import Outcome
from trainparity.results import ProcessEvidence, ResumeResult
from trainparity.serialization import decode_snapshot
from trainparity.snapshot import Snapshot


@dataclass(frozen=True)
class _WorkerResult:
    status: Outcome
    message: str
    snapshots: tuple[Snapshot, ...] = ()
    evidence: ProcessEvidence | None = None


class ResumeRunner:
    """Run an exact A/B resume check across genuine Python process boundaries."""

    def __init__(self, *, timeout: float = 60.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def run(
        self,
        case: str,
        *,
        total_steps: int = 4,
        split_step: int = 2,
        seed: int = 17,
        work_dir: Path | None = None,
    ) -> ResumeResult:
        """Compare continuous and interrupted trajectories under exact policy."""
        if not 0 < split_step < total_steps:
            raise ValueError("split_step must be strictly between zero and total_steps")
        if work_dir is None:
            with tempfile.TemporaryDirectory(prefix="trainparity-") as temporary:
                return self._run(case, total_steps, split_step, seed, Path(temporary))
        work_dir.mkdir(parents=True, exist_ok=True)
        return self._run(case, total_steps, split_step, seed, work_dir)

    def _run(
        self, case: str, total_steps: int, split_step: int, seed: int, work_dir: Path
    ) -> ResumeResult:
        checkpoint = work_dir / "checkpoint.pt"
        common = {"case": case, "seed": seed, "checkpoint": str(checkpoint)}
        baseline_a = self._worker(work_dir, "baseline_a", {**common, "mode": "continuous", "steps": total_steps})
        if baseline_a.status is not Outcome.PASS:
            return self._early(baseline_a, "baseline A")
        baseline_b = self._worker(work_dir, "baseline_b", {**common, "mode": "continuous", "steps": total_steps})
        if baseline_b.status is not Outcome.PASS:
            return self._early(baseline_b, "baseline B", baseline_a=baseline_a.evidence)
        consistency = self._trajectory_difference(baseline_a.snapshots, baseline_b.snapshots)
        if consistency is not None:
            step, differences = consistency
            return ResumeResult(
                Outcome.ABSTAIN,
                f"baseline self-consistency failed at step {step}; resume attribution is unsafe",
                last_matching_step=step - 1,
                first_divergent_step=step,
                phase="completed_training_step",
                primary_difference=differences[0],
                all_differences=differences,
                baseline_a=baseline_a.evidence,
                baseline_b=baseline_b.evidence,
            )

        presave = self._worker(work_dir, "presave", {**common, "mode": "presave", "steps": split_step})
        if presave.status is not Outcome.PASS:
            return self._early(
                presave,
                "pre-save",
                baseline_a=baseline_a.evidence,
                baseline_b=baseline_b.evidence,
            )
        initial = ExactComparison().compare_all(baseline_a.snapshots[0], presave.snapshots[0])
        if initial:
            return ResumeResult(
                Outcome.ABSTAIN,
                "initial snapshots differ; resume attribution is unsafe",
                first_divergent_step=0,
                primary_difference=initial[0],
                all_differences=initial,
                baseline_a=baseline_a.evidence,
                baseline_b=baseline_b.evidence,
                pre_save=presave.evidence,
                checkpoint_path=str(checkpoint),
            )
        resume = self._worker(
            work_dir,
            "resume",
            {**common, "mode": "resume", "steps": total_steps - split_step},
        )
        if resume.status is not Outcome.PASS:
            return self._early(
                resume,
                "post-load",
                baseline_a=baseline_a.evidence,
                baseline_b=baseline_b.evidence,
                pre_save=presave.evidence,
                checkpoint_path=str(checkpoint),
            )
        if presave.evidence is None or resume.evidence is None or presave.evidence.pid == resume.evidence.pid:
            return ResumeResult(
                Outcome.ERROR,
                "pre-save and post-load workers did not prove distinct process IDs",
                baseline_a=baseline_a.evidence,
                baseline_b=baseline_b.evidence,
                pre_save=presave.evidence,
                post_load=resume.evidence,
                checkpoint_path=str(checkpoint),
            )
        candidate = (*presave.snapshots[:split_step], *resume.snapshots)
        divergence = self._trajectory_difference(baseline_a.snapshots, candidate)
        if divergence is None:
            return ResumeResult(
                Outcome.PASS,
                f"trajectories are exactly equivalent through step {total_steps}",
                last_matching_step=total_steps,
                baseline_a=baseline_a.evidence,
                baseline_b=baseline_b.evidence,
                pre_save=presave.evidence,
                post_load=resume.evidence,
                checkpoint_path=str(checkpoint),
            )
        step, differences = divergence
        return ResumeResult(
            Outcome.FAIL,
            f"first observed divergence at step {step}: {differences[0].path}",
            last_matching_step=step - 1,
            first_divergent_step=step,
            phase="completed_training_step",
            primary_difference=differences[0],
            all_differences=differences,
            baseline_a=baseline_a.evidence,
            baseline_b=baseline_b.evidence,
            pre_save=presave.evidence,
            post_load=resume.evidence,
            checkpoint_path=str(checkpoint),
        )

    def _worker(self, work_dir: Path, name: str, request: dict[str, Any]) -> _WorkerResult:
        request_path = work_dir / f"{name}.request.json"
        result_path = work_dir / f"{name}.result.json"
        request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "trainparity.worker",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            return _WorkerResult(Outcome.ERROR, f"worker launch/timeout error: {error}")
        if completed.returncode != 0:
            return _WorkerResult(
                Outcome.ERROR,
                f"worker exited {completed.returncode}: {completed.stderr.strip()}",
            )
        try:
            payload: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
            status = Outcome(payload["status"])
            if status is not Outcome.PASS:
                return _WorkerResult(status, str(payload.get("message", "worker did not pass")))
            snapshots = tuple(decode_snapshot(item) for item in payload["snapshots"])
            evidence = ProcessEvidence(**payload["evidence"])
            return _WorkerResult(status, "worker passed", snapshots, evidence)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return _WorkerResult(Outcome.ERROR, f"worker result corruption: {error}")

    @staticmethod
    def _trajectory_difference(
        baseline: tuple[Snapshot, ...], candidate: tuple[Snapshot, ...]
    ) -> tuple[int, tuple[Any, ...]] | None:
        if len(baseline) != len(candidate):
            raise ValueError("trajectory lengths differ")
        policy = ExactComparison()
        for index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
            differences = policy.compare_all(left, right)
            if differences:
                return index, differences
        return None

    @staticmethod
    def _early(
        worker: _WorkerResult,
        label: str,
        *,
        baseline_a: ProcessEvidence | None = None,
        baseline_b: ProcessEvidence | None = None,
        pre_save: ProcessEvidence | None = None,
        checkpoint_path: str | None = None,
    ) -> ResumeResult:
        outcome = Outcome.ABSTAIN if worker.status is Outcome.ABSTAIN else Outcome.ERROR
        return ResumeResult(
            outcome,
            f"{label} worker: {worker.message}",
            baseline_a=baseline_a,
            baseline_b=baseline_b,
            pre_save=pre_save,
            checkpoint_path=checkpoint_path,
        )

