"""Fresh snapshot worker for command-oriented process resume cases."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from trainparity.importing import load_process_case
from trainparity.outcomes import Outcome
from trainparity.serialization import encode_snapshot
from trainparity.snapshot import Snapshot
from trainparity.state import FrozenMapping, FullValueBackend, UnsupportedStateError


class StableStateError(ValueError):
    """Raised when external checkpoint keys cannot become deterministic paths."""


def _stable_state(value: object) -> object:
    module = type(value).__module__
    if module == "numpy" or module.startswith("numpy."):
        dynamic = cast(Any, value)
        if type(value).__name__ == "ndarray":
            return {
                "dtype": str(dynamic.dtype),
                "shape": list(dynamic.shape),
                "values": dynamic.tolist(),
            }
        return dynamic.item()
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, child in value.items():
            stable_key = str(key)
            if stable_key in converted:
                raise StableStateError(
                    f"checkpoint key collision after string conversion: {stable_key!r}"
                )
            converted[stable_key] = _stable_state(child)
        return converted
    if isinstance(value, list):
        return [_stable_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stable_state(item) for item in value)
    return value


def execute(case_spec: str, checkpoint: Path, step: int) -> dict[str, Any]:
    """Load, freeze, and serialize one external checkpoint without mutable aliases."""
    try:
        case = load_process_case(case_spec)
    except Exception as error:
        return {
            "outcome": Outcome.ERROR.value,
            "message": f"process case loading failed: {type(error).__name__}",
        }
    try:
        capture_started = time.perf_counter()
        observed = case.observe_checkpoint(checkpoint)
    except Exception as error:
        return {
            "outcome": Outcome.ERROR.value,
            "message": f"observe_checkpoint callback failed: {type(error).__name__}",
        }
    try:
        frozen = FullValueBackend().freeze(_stable_state(observed))
        if not isinstance(frozen, FrozenMapping):
            raise StableStateError("observed checkpoint root must be a mapping")
        capture_seconds = time.perf_counter() - capture_started
        serialization_started = time.perf_counter()
        encoded = encode_snapshot(Snapshot(step, frozen))
        serialization_seconds = time.perf_counter() - serialization_started
        return {
            "outcome": Outcome.PASS.value,
            "message": "snapshot captured",
            "snapshot": encoded,
            "timing_seconds": {
                "capture": capture_seconds,
                "serialization": serialization_seconds,
            },
        }
    except (StableStateError, UnsupportedStateError) as error:
        return {
            "outcome": Outcome.ABSTAIN.value,
            "message": f"snapshot state is unsupported: {type(error).__name__}",
        }
    except Exception as error:
        return {
            "outcome": Outcome.ERROR.value,
            "message": f"snapshot worker failed: {type(error).__name__}",
        }


def main() -> int:
    """Capture a snapshot and atomically publish its deterministic JSON envelope."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.case, args.checkpoint, args.step)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    temporary.replace(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
