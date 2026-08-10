"""Fresh-process worker used by the Gate 3 orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

from trainparity.importing import load_case
from trainparity.outcomes import Outcome
from trainparity.protocols import ResumeExecutionCase, StepObservation, TrainingState
from trainparity.serialization import encode_snapshot
from trainparity.snapshot import CaptureResult, capture_snapshot


def _evidence(state: TrainingState) -> dict[str, int | None]:
    return {
        "pid": os.getpid(),
        "model_id": id(state.model),
        "optimizer_id": id(state.optimizer),
        "scheduler_id": None if state.scheduler is None else id(state.scheduler),
        "scaler_id": None if state.scaler is None else id(state.scaler),
    }


def _capture(state: TrainingState, observation: StepObservation | None) -> CaptureResult:
    extras = None if observation is None else observation.extras
    batch = None if observation is None else observation.batch_state()
    return capture_snapshot(
        state.model,
        step=state.step,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        scaler=state.scaler,
        extras=extras,
        batch=batch,
    )


def _append_snapshot(
    snapshots: list[dict[str, Any]], state: TrainingState, observation: StepObservation | None
) -> dict[str, Any] | None:
    captured = _capture(state, observation)
    if captured.outcome is not Outcome.PASS or captured.snapshot is None:
        issue = captured.issue
        return {
            "status": captured.outcome.value,
            "message": "snapshot capture did not complete",
            "issue": None if issue is None else {"path": issue.path, "detail": issue.detail},
        }
    snapshots.append(encode_snapshot(captured.snapshot))
    return None


def execute(request: dict[str, Any]) -> dict[str, Any]:
    """Execute one worker request; called by CLI and unit tests."""
    case = load_case(str(request["case"]))
    if not isinstance(case, ResumeExecutionCase):
        raise TypeError("Gate 3 case must implement ResumeExecutionCase")
    mode = str(request["mode"])
    seed = int(request["seed"])
    steps = int(request["steps"])
    checkpoint = Path(str(request["checkpoint"]))
    if mode == "resume":
        state = case.load(checkpoint, seed)
    elif mode in {"continuous", "presave"}:
        state = case.build(seed)
    else:
        raise ValueError(f"unknown worker mode: {mode}")

    snapshots: list[dict[str, Any]] = []
    initial_observation = case.observe(state) if mode == "resume" else None
    problem = _append_snapshot(snapshots, state, initial_observation)
    if problem is not None:
        return problem

    for _ in range(steps):
        case.train_step(state)
        observation = case.observe(state)
        if observation.batch_state() is None:
            return {
                "status": Outcome.ABSTAIN.value,
                "message": "stable batch identity is unavailable after a completed step",
            }
        problem = _append_snapshot(snapshots, state, observation)
        if problem is not None:
            return problem

    if mode == "presave":
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        case.save(state, checkpoint)
        if not checkpoint.is_file():
            raise OSError(f"case did not create checkpoint file: {checkpoint}")
    return {
        "status": Outcome.PASS.value,
        "evidence": _evidence(state),
        "snapshots": snapshots,
    }


def main() -> int:
    """Read a request and atomically publish a JSON result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        request: dict[str, Any] = json.loads(args.request.read_text(encoding="utf-8"))
        result = execute(request)
    except Exception as error:
        result = {
            "status": Outcome.ERROR.value,
            "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    temporary.replace(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

