"""Deterministic JSON serialization for snapshots crossing worker boundaries."""

from __future__ import annotations

import base64
import struct
from typing import Any, cast

from trainparity.snapshot import Snapshot
from trainparity.state import FrozenMapping, FrozenSequence, FrozenTensor, FrozenValue


def encode_snapshot(snapshot: Snapshot) -> dict[str, Any]:
    """Encode a snapshot without pickle or mutable tensor aliases."""
    return {
        "schema_version": snapshot.schema_version,
        "backend": snapshot.backend,
        "step": snapshot.step,
        "state": _encode_value(snapshot.state),
    }


def decode_snapshot(payload: dict[str, Any]) -> Snapshot:
    """Decode and validate a worker-produced snapshot payload."""
    state = _decode_value(payload["state"])
    if not isinstance(state, FrozenMapping):
        raise ValueError("snapshot state must decode to FrozenMapping")
    return Snapshot(
        step=payload["step"],
        state=state,
        backend=payload["backend"],
        schema_version=payload["schema_version"],
    )


def _encode_value(value: FrozenValue) -> dict[str, Any]:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": str(value)}
    if isinstance(value, float):
        bits = base64.b64encode(struct.pack(">d", value)).decode("ascii")
        return {"kind": "float", "bits": bits}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {"kind": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, FrozenTensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": value.dtype,
            "device": value.device,
            "requires_grad": value.requires_grad,
            "data": base64.b64encode(value.data).decode("ascii"),
        }
    if isinstance(value, FrozenMapping):
        return {
            "kind": "mapping",
            "entries": [[key, _encode_value(child)] for key, child in value.entries],
        }
    if isinstance(value, FrozenSequence):
        return {
            "kind": "sequence",
            "sequence_kind": value.kind,
            "items": [_encode_value(child) for child in value.items],
        }
    raise TypeError(f"cannot encode frozen value {type(value)!r}")


def _decode_value(payload: dict[str, Any]) -> FrozenValue:
    kind = payload.get("kind")
    if kind == "none":
        return None
    if kind == "bool":
        return bool(payload["value"])
    if kind == "int":
        return int(payload["value"])
    if kind == "float":
        return cast(float, struct.unpack(">d", base64.b64decode(payload["bits"], validate=True))[0])
    if kind == "str":
        return str(payload["value"])
    if kind == "bytes":
        return base64.b64decode(payload["data"], validate=True)
    if kind == "tensor":
        return FrozenTensor(
            shape=tuple(int(value) for value in payload["shape"]),
            dtype=str(payload["dtype"]),
            device=str(payload["device"]),
            requires_grad=bool(payload["requires_grad"]),
            data=base64.b64decode(payload["data"], validate=True),
        )
    if kind == "mapping":
        entries = payload["entries"]
        if not isinstance(entries, list):
            raise ValueError("mapping entries must be a list")
        return FrozenMapping(tuple((str(key), _decode_value(value)) for key, value in entries))
    if kind == "sequence":
        return FrozenSequence(
            str(payload["sequence_kind"]),
            tuple(_decode_value(value) for value in payload["items"]),
        )
    raise ValueError(f"unknown frozen value kind: {kind!r}")
