from __future__ import annotations

import random
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from trainparity import ExactComparison, Outcome, capture_snapshot
from trainparity.snapshot import Snapshot
from trainparity.state import FrozenMapping, FrozenTensor, FrozenValue


class ModelWithBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 1)
        self.register_buffer("count", torch.tensor(2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer(value)


class ExtraState:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> Mapping[str, object]:
        return {"nested": {"value": self.value}}


def require_snapshot(result) -> Snapshot:
    assert result.outcome is Outcome.PASS, result.issue
    assert result.snapshot is not None
    return result.snapshot


def flatten(value: FrozenValue, prefix: str = "") -> dict[str, FrozenValue]:
    if isinstance(value, FrozenMapping):
        result: dict[str, FrozenValue] = {}
        for key, child in value.entries:
            path = f"{prefix}.{key}" if prefix else key
            result.update(flatten(child, path))
        return result
    return {prefix: value}


def test_capture_includes_model_buffer_gradient_rng_and_extra_state() -> None:
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    model = ModelWithBuffer()
    model(torch.ones(1, 2)).sum().backward()
    captured = require_snapshot(
        capture_snapshot(model, step=3, extras={"ema": ExtraState(4)}, capture_rng=True)
    )
    paths = flatten(captured.state)
    assert captured.step == 3
    assert isinstance(paths["model.layer.weight"], FrozenTensor)
    assert isinstance(paths["buffer.count"], FrozenTensor)
    assert isinstance(paths["gradient.layer.weight"], FrozenTensor)
    assert paths["extra.ema.nested.value"] == 4
    assert "rng.python[0]" not in paths  # sequences stay one branch in this helper
    assert "rng.torch_cpu" in paths
    assert "rng.numpy.algorithm" in paths


def test_capture_freezes_model_and_extra_tensor_aliases() -> None:
    model = ModelWithBuffer()
    extra = torch.tensor([5.0])
    captured = require_snapshot(capture_snapshot(model, extras={"tensor": extra}, capture_rng=False))
    before = flatten(captured.state)
    model.layer.weight.data.add_(10)
    extra.add_(10)
    after = flatten(captured.state)
    assert before == after
    assert before["extra.tensor"].to_tensor().item() == 5.0  # type: ignore[union-attr]


def test_missing_none_and_empty_sections_are_distinct() -> None:
    model = ModelWithBuffer()
    missing = require_snapshot(capture_snapshot(model, capture_rng=False))
    explicit_none = require_snapshot(capture_snapshot(model, scheduler=None, capture_rng=False))
    empty = require_snapshot(capture_snapshot(model, extras={}, capture_rng=False))
    none_difference = ExactComparison().compare(missing, explicit_none).first_difference
    empty_difference = ExactComparison().compare(missing, empty).first_difference
    assert none_difference is not None and none_difference.path == "scheduler"
    assert empty_difference is not None and empty_difference.path == "extra"


def test_unsupported_state_abstains_and_state_dict_crash_errors() -> None:
    model = ModelWithBuffer()
    unsupported = capture_snapshot(model, extras={"bad": object()}, capture_rng=False)
    assert unsupported.outcome is Outcome.ABSTAIN
    assert unsupported.issue is not None and unsupported.issue.path == "extra.bad"

    class Broken:
        def state_dict(self) -> Mapping[str, object]:
            raise RuntimeError("boom")

    broken = capture_snapshot(model, extras={"broken": Broken()}, capture_rng=False)
    assert broken.outcome is Outcome.ERROR
    assert broken.issue is not None and "boom" in broken.issue.detail


def test_sparse_state_abstains() -> None:
    model = ModelWithBuffer()
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0]]), torch.tensor([1.0]), (2,), check_invariants=True
    )
    result = capture_snapshot(model, extras={"sparse": sparse}, capture_rng=False)
    assert result.outcome is Outcome.ABSTAIN


def test_scheduler_and_scaler_state_are_captured() -> None:
    model = ModelWithBuffer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    class Scaler:
        def state_dict(self) -> Mapping[str, object]:
            return {"scale": 16.0}

    captured = require_snapshot(
        capture_snapshot(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=Scaler(),
            capture_rng=False,
        )
    )
    paths = flatten(captured.state)
    assert paths["scheduler.last_epoch"] == 0
    assert paths["scaler.scale"] == 16.0


def test_custom_backend_boundary_is_supported() -> None:
    class Backend:
        name = "test_backend"

        def freeze(self, value: object, path=()) -> object:
            return FrozenMapping((("sentinel", 1),))

    captured = require_snapshot(capture_snapshot(ModelWithBuffer(), backend=Backend()))
    assert captured.backend == "test_backend"
    assert captured.state == FrozenMapping((("sentinel", 1),))


def test_backend_with_invalid_root_errors() -> None:
    class Backend:
        name = "invalid"

        def freeze(self, value: object, path=()) -> object:
            return 1

    result = capture_snapshot(ModelWithBuffer(), backend=Backend())
    assert result.outcome is Outcome.ERROR
