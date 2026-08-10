"""Immutable full-value state representation and deterministic paths."""

from __future__ import annotations

import ctypes
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import torch

PathPart: TypeAlias = str | int
StatePath: TypeAlias = tuple[PathPart, ...]


class UnsupportedStateError(TypeError):
    """Raised when the reference backend cannot freeze a value safely."""

    def __init__(self, path: StatePath, value: object) -> None:
        self.path = path
        self.value_type = f"{type(value).__module__}.{type(value).__qualname__}"
        super().__init__(f"unsupported state at {render_path(path)}: {self.value_type}")


class AmbiguousStateNameError(ValueError):
    """Raised when dotted names cannot form one unambiguous tree."""


@dataclass(frozen=True)
class FrozenTensor:
    """Immutable tensor bytes plus original comparison metadata."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    data: bytes

    @classmethod
    def capture(cls, value: torch.Tensor) -> FrozenTensor:
        """Clone tensor storage to CPU bytes without retaining a mutable alias."""
        if value.layout != torch.strided:
            raise ValueError(f"unsupported tensor layout: {value.layout}")
        original_device = str(value.device)
        frozen = value.detach().cpu().contiguous().clone()
        byte_count = frozen.numel() * frozen.element_size()
        data = b"" if byte_count == 0 else ctypes.string_at(frozen.data_ptr(), byte_count)
        return cls(
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            device=original_device,
            requires_grad=value.requires_grad,
            data=data,
        )

    def to_tensor(self) -> torch.Tensor:
        """Materialize an independent CPU tensor for reference comparison."""
        dtype = getattr(torch, self.dtype.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"unknown captured dtype: {self.dtype}")
        if not self.data:
            return torch.empty(self.shape, dtype=dtype)
        return torch.frombuffer(bytearray(self.data), dtype=dtype).reshape(self.shape).clone()


@dataclass(frozen=True)
class FrozenMapping:
    """String-keyed mapping stored in lexical key order."""

    entries: tuple[tuple[str, FrozenValue], ...]


@dataclass(frozen=True)
class FrozenSequence:
    """Sequence that preserves whether the source was a list or tuple."""

    kind: str
    items: tuple[FrozenValue, ...]


FrozenScalar: TypeAlias = bool | int | float | str | bytes | None
FrozenValue: TypeAlias = FrozenScalar | FrozenTensor | FrozenMapping | FrozenSequence


class FullValueBackend:
    """Gate 2 reference backend that materializes complete immutable values."""

    name = "full_value_reference"

    def freeze(self, value: object, path: StatePath = ()) -> FrozenValue:
        """Recursively freeze a supported Python/PyTorch state value."""
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        if isinstance(value, torch.Tensor):
            try:
                return FrozenTensor.capture(value)
            except ValueError as error:
                raise UnsupportedStateError(path, value) from error
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise UnsupportedStateError(path, value)
            return FrozenMapping(
                tuple(
                    (key, self.freeze(value[key], (*path, key)))
                    for key in sorted(value)
                )
            )
        if isinstance(value, (list, tuple)):
            kind = "list" if isinstance(value, list) else "tuple"
            return FrozenSequence(
                kind,
                tuple(self.freeze(item, (*path, index)) for index, item in enumerate(value)),
            )
        raise UnsupportedStateError(path, value)


def render_path(path: StatePath) -> str:
    """Render an unambiguous deterministic state path."""
    if not path:
        return "$"
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif part.isidentifier():
            rendered += ("." if rendered else "") + part
        else:
            rendered += f"[{json.dumps(part, ensure_ascii=False)}]"
    return rendered


def nested_named_values(values: Mapping[str, object]) -> dict[str, object]:
    """Expand dotted PyTorch names into deterministic nested mappings."""
    root: dict[str, object] = {}
    for name in sorted(values):
        parts = name.split(".")
        current = root
        for part in parts[:-1]:
            existing = current.setdefault(part, {})
            if not isinstance(existing, dict):
                raise AmbiguousStateNameError(f"ambiguous dotted state name: {name}")
            current = existing
        if parts[-1] in current:
            raise AmbiguousStateNameError(f"duplicate state name: {name}")
        current[parts[-1]] = values[name]
    return root
