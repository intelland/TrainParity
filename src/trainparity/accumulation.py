"""Bounded fresh-process accumulation equivalence orchestration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from trainparity.comparison import ExactComparison, ToleranceComparison
from trainparity.importing import CaseImportError, load_accumulation_case
from trainparity.outcomes import Outcome
from trainparity.results import AccumulationResult
from trainparity.serialization import decode_snapshot
from trainparity.snapshot import Snapshot

PHASES = (
    "loss_accounting",
    "gradient",
    "optimizer_state",
    "parameter_update",
    "scheduler_state",
)


class UnsafeBatchSplit(ValueError):
    """The default splitter cannot preserve the declared batch semantics."""


@dataclass(frozen=True)
class AccumulationExecutionPlan:
    """Framework-neutral actions inside exactly one optimizer-update window."""

    microbatch_count: int
    scale_accumulated_loss: bool = True
    use_explicit_loss_accounting: bool = True
    optimizer_step_per_microbatch: bool = False
    scheduler_step_per_microbatch: bool = False
    zero_grad_before_gradient_observation: bool = False
    clip_grad_norm: float | None = None
    clip_per_microbatch: bool = False
    amp_step_before_unscale: bool = False
    omit_final_microbatch: bool = False

    def validate(self) -> None:
        """Reject ambiguous plans before child execution."""
        if self.microbatch_count < 1:
            raise ValueError("microbatch_count must be positive")
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError("clip_grad_norm must be positive")

    def to_dict(self) -> dict[str, object]:
        """Serialize a plan for the fresh worker."""
        return {
            "microbatch_count": self.microbatch_count,
            "scale_accumulated_loss": self.scale_accumulated_loss,
            "use_explicit_loss_accounting": self.use_explicit_loss_accounting,
            "optimizer_step_per_microbatch": self.optimizer_step_per_microbatch,
            "scheduler_step_per_microbatch": self.scheduler_step_per_microbatch,
            "zero_grad_before_gradient_observation": self.zero_grad_before_gradient_observation,
            "clip_grad_norm": self.clip_grad_norm,
            "clip_per_microbatch": self.clip_per_microbatch,
            "amp_step_before_unscale": self.amp_step_before_unscale,
            "omit_final_microbatch": self.omit_final_microbatch,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AccumulationExecutionPlan:
        """Decode a worker plan without accepting unknown fields."""
        allowed = {field.name for field in __import__("dataclasses").fields(cls)}
        if set(value) != allowed:
            raise ValueError("accumulation plan fields are incomplete or unknown")
        plan = cls(**value)  # type: ignore[arg-type]
        plan.validate()
        return plan


def split_tensor_tree(batch: object, parts: int) -> tuple[object, ...]:
    """Split a supported tensor tree on dim 0 while preserving structure/order."""
    if parts < 1:
        raise UnsafeBatchSplit("parts must be positive")
    length = _tree_batch_length(batch)
    if length < parts:
        raise UnsafeBatchSplit("microbatch count exceeds batch length")
    indices = tuple(torch.tensor_split(torch.arange(length), parts))
    return tuple(_slice_tree(batch, index) for index in indices)


def _tree_batch_length(value: object) -> int:
    lengths: list[int] = []

    def visit(node: object) -> None:
        if isinstance(node, torch.Tensor):
            if node.ndim == 0:
                raise UnsafeBatchSplit("scalar tensor leaf has no batch dimension")
            lengths.append(len(node))
        elif isinstance(node, Mapping):
            if not node or not all(isinstance(key, str) for key in node):
                raise UnsafeBatchSplit("mapping leaves must be non-empty and string-keyed")
            for child in node.values():
                visit(child)
        elif isinstance(node, (tuple, list)):
            if not node:
                raise UnsafeBatchSplit("sequence batch must be non-empty")
            for child in node:
                visit(child)
        else:
            raise UnsafeBatchSplit(f"unsupported batch leaf: {type(node).__name__}")

    visit(value)
    if not lengths or lengths[0] == 0 or len(set(lengths)) != 1:
        raise UnsafeBatchSplit("tensor leaves must have one equal non-zero leading length")
    return lengths[0]


def _slice_tree(value: object, index: torch.Tensor) -> object:
    if isinstance(value, torch.Tensor):
        return value[index.to(value.device)]
    if isinstance(value, Mapping):
        return {key: _slice_tree(child, index) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_slice_tree(child, index) for child in value)
    if isinstance(value, list):
        return [_slice_tree(child, index) for child in value]
    raise UnsafeBatchSplit(f"unsupported batch leaf: {type(value).__name__}")


@dataclass(frozen=True)
class _WorkerRecord:
    outcome: Outcome
    message: str
    pid: int
    equivalence: str
    initial: Snapshot | None
    phases: dict[str, Snapshot]
    loss_captured: bool


class AccumulationRunner:
    """Compare one full-batch update with a declared accumulated update."""

    def __init__(
        self,
        *,
        comparison: ExactComparison | ToleranceComparison,
        timeout: float = 300.0,
        temporary_root: Path | None = None,
    ) -> None:
        if not isinstance(comparison, (ExactComparison, ToleranceComparison)):
            raise TypeError("comparison must be explicit ExactComparison or ToleranceComparison")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.comparison = comparison
        self.timeout = timeout
        self.temporary_root = temporary_root

    def run(
        self,
        case: str,
        *,
        candidate: AccumulationExecutionPlan,
        device: str = "cpu",
        seed: int = 23,
        cwd: Path | None = None,
        report_path: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> AccumulationResult:
        """Run two full-batch controls and one candidate in fresh processes."""
        baseline = AccumulationExecutionPlan(
            1,
            clip_grad_norm=candidate.clip_grad_norm,
        )
        try:
            candidate.validate()
            selected = load_accumulation_case(case)
            if not selected.equivalence.strip():
                raise ValueError("case must declare a non-empty equivalence relation")
        except (CaseImportError, ValueError) as error:
            result = AccumulationResult(
                Outcome.ERROR,
                f"case setup failed: {type(error).__name__}: {error}",
                case,
                "",
            )
            return self._finish(result, report_path)
        root_parent = self.temporary_root
        if root_parent is not None:
            root_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="trainparity-accum-", dir=root_parent) as raw:
            root = Path(raw)
            records: list[_WorkerRecord] = []
            peak = 0
            for name, plan in (("baseline_a", baseline), ("baseline_b", baseline), ("candidate", candidate)):
                record, observed_peak = self._worker(
                    case, plan, device, seed, root / name, Path(cwd or Path.cwd()),
                    dict(environment or {}), root,
                )
                peak = max(peak, observed_peak)
                records.append(record)
                if record.outcome is not Outcome.PASS:
                    return self._finish(
                        AccumulationResult(
                            record.outcome, record.message, case, selected.equivalence,
                            process_ids=tuple(item.pid for item in records),
                            comparison_policy=self._policy_name(),
                            peak_temporary_directory_bytes=peak,
                        ), report_path,
                    )
            first, second, candidate_record = records
            if len({item.pid for item in records}) != 3:
                result = AccumulationResult(
                    Outcome.ERROR, "fresh executions did not have distinct process IDs", case,
                    selected.equivalence, process_ids=tuple(item.pid for item in records),
                    comparison_policy=self._policy_name(), peak_temporary_directory_bytes=peak,
                )
                return self._finish(result, report_path)
            initial_problem = self._first_difference(first.initial, second.initial)
            if initial_problem is not None:
                result = self._divergence(
                    Outcome.ABSTAIN, "initial_state", initial_problem, records, case,
                    selected.equivalence, peak, False,
                    "baseline initial-state self-consistency failed",
                )
                return self._finish(result, report_path)
            initial_problem = self._first_difference(first.initial, candidate_record.initial)
            if initial_problem is not None:
                result = self._divergence(
                    Outcome.ABSTAIN, "initial_state", initial_problem, records, case,
                    selected.equivalence, peak, False,
                    "candidate did not start from verified-equal initial state",
                )
                return self._finish(result, report_path)
            for phase in PHASES:
                control = self.comparison.compare_all(first.phases[phase], second.phases[phase])
                if control:
                    result = self._divergence(
                        Outcome.ABSTAIN, phase, control, records, case, selected.equivalence,
                        peak, True, "baseline self-consistency failed",
                    )
                    return self._finish(result, report_path)
                differences = self.comparison.compare_all(
                    first.phases[phase], candidate_record.phases[phase]
                )
                if differences:
                    result = self._divergence(
                        Outcome.FAIL, phase, differences, records, case, selected.equivalence,
                        peak, True, f"first observed divergence in {phase}; not a root-cause claim",
                    )
                    return self._finish(result, report_path)
            result = AccumulationResult(
                Outcome.PASS,
                "declared executions are equivalent for one optimizer-update boundary",
                case,
                selected.equivalence,
                process_ids=tuple(item.pid for item in records),
                verified_equal_initial_state=True,
                loss_normalization_captured=all(item.loss_captured for item in records),
                comparison_policy=self._policy_name(),
                peak_temporary_directory_bytes=peak,
            )
            return self._finish(result, report_path)

    def _worker(
        self,
        case: str,
        plan: AccumulationExecutionPlan,
        device: str,
        seed: int,
        output_dir: Path,
        cwd: Path,
        updates: dict[str, str],
        temp_root: Path,
    ) -> tuple[_WorkerRecord, int]:
        output_dir.mkdir(parents=True)
        result_path = output_dir / "result.json"
        command = [
            sys.executable, "-m", "trainparity.accumulation_worker", "--case", case,
            "--plan", json.dumps(plan.to_dict(), sort_keys=True), "--device", device,
            "--seed", str(seed), "--result", str(result_path),
        ]
        environment = os.environ.copy()
        environment.update(updates)
        process = subprocess.Popen(command, cwd=cwd, env=environment)
        deadline = time.monotonic() + self.timeout
        peak = 0
        while process.poll() is None:
            peak = max(peak, _tree_size(temp_root))
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                return _WorkerRecord(Outcome.ERROR, "worker timeout", process.pid, "", None, {}, False), peak
            time.sleep(0.01)
        peak = max(peak, _tree_size(temp_root))
        if process.returncode != 0 or not result_path.is_file():
            return _WorkerRecord(Outcome.ERROR, "worker did not publish a result", process.pid, "", None, {}, False), peak
        try:
            payload: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
            outcome = Outcome(payload["outcome"])
            initial = decode_snapshot(payload["initial"]) if payload.get("initial") else None
            phases = {name: decode_snapshot(value) for name, value in payload.get("phases", {}).items()}
            return _WorkerRecord(
                outcome, str(payload["message"]), int(payload["pid"]),
                str(payload.get("equivalence", "")), initial, phases,
                bool(payload.get("loss_normalization_captured", False)),
            ), peak
        except (OSError, ValueError, KeyError, TypeError):
            return _WorkerRecord(Outcome.ERROR, "worker result was corrupt", process.pid, "", None, {}, False), peak

    def _first_difference(
        self, left: Snapshot | None, right: Snapshot | None
    ) -> tuple[Any, ...] | None:
        if left is None or right is None:
            return ()
        differences = self.comparison.compare_all(left, right)
        return differences or None

    def _divergence(
        self, outcome: Outcome, phase: str, differences: Sequence[Any],
        records: Sequence[_WorkerRecord], case: str, equivalence: str, peak: int,
        initial_equal: bool, message: str,
    ) -> AccumulationResult:
        return AccumulationResult(
            outcome, message, case, equivalence, phase,
            None if not differences else differences[0], tuple(differences),
            tuple(item.pid for item in records), initial_equal,
            all(item.loss_captured for item in records), self._policy_name(), peak,
        )

    def _policy_name(self) -> str:
        return "exact" if isinstance(self.comparison, ExactComparison) else "explicit_tolerance"

    @staticmethod
    def _finish(result: AccumulationResult, report_path: Path | None) -> AccumulationResult:
        if report_path is None:
            return result
        report_path.parent.mkdir(parents=True, exist_ok=True)
        finished = result
        for _ in range(8):
            payload = (json.dumps(finished.to_dict(), indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
            observed_size = len(payload)
            if observed_size == finished.persisted_artifact_bytes:
                break
            finished = replace(finished, persisted_artifact_bytes=observed_size)
        else:
            raise RuntimeError("accumulation report byte size did not stabilize")
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_bytes(payload)
        if temporary.stat().st_size != len(payload):
            raise OSError("temporary accumulation report byte size changed")
        temporary.replace(report_path)
        if report_path.stat().st_size != len(payload):
            raise OSError("published accumulation report byte size changed")
        return finished


def _tree_size(root: Path) -> int:
    try:
        return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    except OSError:
        return 0
