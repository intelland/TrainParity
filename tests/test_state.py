from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from trainparity.state import (
    AmbiguousStateNameError,
    FrozenMapping,
    FrozenSequence,
    FrozenTensor,
    FullValueBackend,
    UnsupportedStateError,
    nested_named_values,
    render_path,
)


def test_paths_are_stable_and_escape_ambiguous_keys() -> None:
    assert render_path(()) == "$"
    assert render_path(("model", "layer", 2, "weight")) == "model.layer[2].weight"
    assert render_path(("extra", "a.b", "not valid")) == 'extra["a.b"]["not valid"]'


def test_full_value_backend_sorts_mappings_and_preserves_sequence_kind() -> None:
    frozen = FullValueBackend().freeze({"z": (1, 2), "a": [3]})
    assert isinstance(frozen, FrozenMapping)
    assert [key for key, _ in frozen.entries] == ["a", "z"]
    assert isinstance(frozen.entries[0][1], FrozenSequence)
    assert frozen.entries[0][1].kind == "list"
    assert isinstance(frozen.entries[1][1], FrozenSequence)
    assert frozen.entries[1][1].kind == "tuple"


def test_tensor_capture_breaks_mutable_alias_and_round_trips() -> None:
    source = torch.tensor([1.0, -0.0], requires_grad=True)
    frozen = FrozenTensor.capture(source)
    source.data.add_(9)
    restored = frozen.to_tensor()
    assert restored.tolist() == [1.0, -0.0]
    assert frozen.requires_grad is True
    restored.add_(5)
    assert frozen.to_tensor().tolist() == [1.0, -0.0]


def test_empty_tensor_and_dtype_metadata_round_trip() -> None:
    frozen = FrozenTensor.capture(torch.empty((0, 2), dtype=torch.int64))
    restored = frozen.to_tensor()
    assert restored.shape == (0, 2)
    assert restored.dtype == torch.int64
    assert frozen.data == b""


def test_unknown_dtype_and_unsupported_values_are_explicit() -> None:
    frozen = FrozenTensor.capture(torch.tensor([1]))
    with pytest.raises(TypeError, match="unknown captured dtype"):
        replace(frozen, dtype="torch.unknown").to_tensor()
    with pytest.raises(UnsupportedStateError, match="unsupported state"):
        FullValueBackend().freeze({"bad": object()})
    with pytest.raises(UnsupportedStateError):
        FullValueBackend().freeze({1: "non-string key"})


def test_sparse_tensor_is_not_silently_materialized() -> None:
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0]]), torch.tensor([1.0]), (2,), check_invariants=True
    )
    with pytest.raises(UnsupportedStateError):
        FullValueBackend().freeze(sparse)


def test_dotted_name_expansion_rejects_collisions() -> None:
    assert nested_named_values({"layer.weight": 1}) == {"layer": {"weight": 1}}
    with pytest.raises(AmbiguousStateNameError):
        nested_named_values({"layer": 1, "layer.weight": 2})
