"""Profile the accepted Gate 4 ImageNet snapshot path before optimization."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from experiments.gate4.adapters.pytorch_examples_imagenet import ImageNetAdapter
from trainparity.serialization import encode_snapshot
from trainparity.snapshot import Snapshot
from trainparity.state import FrozenMapping, FullValueBackend

T = TypeVar("T")


def _stable_keys(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _stable_keys(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_stable_keys(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stable_keys(item) for item in value)
    return value


def _profile(call: Callable[[], T]) -> tuple[T, float, str]:
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    value = call()
    profiler.disable()
    elapsed = time.perf_counter() - started
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)
    return value, elapsed, stream.getvalue()


def run(checkpoint: Path, output: Path) -> dict[str, Any]:
    adapter = ImageNetAdapter()
    raw, load_seconds, load_profile = _profile(lambda: _stable_keys(adapter.normalize_checkpoint(checkpoint)))
    frozen, freeze_seconds, freeze_profile = _profile(lambda: FullValueBackend().freeze(raw))
    if not isinstance(frozen, FrozenMapping):
        raise TypeError("profiled checkpoint did not produce a mapping")
    snapshot = Snapshot(3, frozen)
    payload, serialization_seconds, serialization_profile = _profile(
        lambda: json.dumps(encode_snapshot(snapshot), sort_keys=True).encode("utf-8")
    )
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "snapshot_bytes": len(payload),
        "timing_seconds": {
            "checkpoint_load_and_stable_keys": load_seconds,
            "full_value_freeze": freeze_seconds,
            "json_encode": serialization_seconds,
            "total_snapshot_path": load_seconds + freeze_seconds + serialization_seconds,
        },
        "profiles": {
            "checkpoint_load_and_stable_keys": load_profile,
            "full_value_freeze": freeze_profile,
            "json_encode": serialization_profile,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.checkpoint.resolve(), args.output.resolve())
    print(json.dumps(report["timing_seconds"], sort_keys=True))


if __name__ == "__main__":
    main()
