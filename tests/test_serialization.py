from __future__ import annotations

import math

import torch

from trainparity.comparison import ExactComparison
from trainparity.serialization import decode_snapshot, encode_snapshot
from trainparity.snapshot import capture_snapshot


def test_snapshot_json_round_trip_preserves_exact_values() -> None:
    model = torch.nn.Linear(1, 1)
    captured = capture_snapshot(
        model,
        step=2,
        extras={"values": [None, b"bytes", float("nan"), float("inf"), torch.tensor([])]},
    )
    assert captured.snapshot is not None
    restored = decode_snapshot(encode_snapshot(captured.snapshot))
    assert ExactComparison().compare(captured.snapshot, restored).outcome == "PASS"


def test_snapshot_decoder_rejects_unknown_values() -> None:
    payload = {"schema_version": 1, "backend": "x", "step": 0, "state": {"kind": "bad"}}
    try:
        decode_snapshot(payload)
    except ValueError as error:
        assert "unknown" in str(error)
    else:  # pragma: no cover
        raise AssertionError("invalid payload accepted")


def test_float_round_trip_keeps_signed_zero_and_nan() -> None:
    model = torch.nn.Linear(1, 1)
    captured = capture_snapshot(model, extras={"negative_zero": -0.0, "nan": math.nan})
    assert captured.snapshot is not None
    restored = decode_snapshot(encode_snapshot(captured.snapshot))
    assert ExactComparison().compare(captured.snapshot, restored).outcome == "PASS"

