"""Gate 2 snapshot schema and full-value capture implementation."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.optim import Optimizer

from trainparity.optimizer_state import OptimizerMappingError, canonicalize_optimizer
from trainparity.outcomes import Outcome
from trainparity.state import (
    AmbiguousStateNameError,
    FrozenMapping,
    FullValueBackend,
    StatePath,
    UnsupportedStateError,
    nested_named_values,
    render_path,
)


class Stateful(Protocol):
    """Object exposing serializable state through PyTorch convention."""

    def state_dict(self) -> Mapping[str, object]:
        """Return current state."""


class SnapshotBackend(Protocol):
    """Replaceable backend boundary for snapshot storage."""

    name: str

    def freeze(self, value: object, path: StatePath = ()) -> object:
        """Freeze a supported value."""


class _Missing:
    pass


MISSING = _Missing()


@dataclass(frozen=True)
class Snapshot:
    """One deterministic observation captured at a logical step boundary."""

    step: int | None
    state: FrozenMapping
    backend: str = "full_value_reference"
    schema_version: int = 1


@dataclass(frozen=True)
class CaptureIssue:
    """A precise reason why capture abstained or errored."""

    path: str
    detail: str


@dataclass(frozen=True)
class CaptureResult:
    """Four-state snapshot capture result."""

    outcome: Outcome
    snapshot: Snapshot | None = None
    issue: CaptureIssue | None = None


def _numpy_rng_state() -> object:
    try:
        import numpy as np
    except ImportError:
        return MISSING
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "algorithm": name,
        "keys": {"dtype": str(keys.dtype), "shape": list(keys.shape), "values": keys.tolist()},
        "position": position,
        "has_gauss": has_gauss,
        "cached_gaussian": cached,
    }


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    numpy_state = _numpy_rng_state()
    if numpy_state is not MISSING:
        state["numpy"] = numpy_state
    if torch.cuda.is_available():
        state["torch_cuda"] = {
            f"device_{index}": value
            for index, value in enumerate(torch.cuda.get_rng_state_all())
        }
    return state


def _stateful_value(value: Stateful | None) -> object:
    return None if value is None else dict(value.state_dict())


def capture_snapshot(
    model: nn.Module,
    *,
    step: int | None = None,
    optimizer: Optimizer | None | _Missing = MISSING,
    scheduler: Stateful | None | _Missing = MISSING,
    scaler: Stateful | None | _Missing = MISSING,
    extras: Mapping[str, object] | None = None,
    capture_rng: bool = True,
    backend: SnapshotBackend | None = None,
) -> CaptureResult:
    """Capture supported training state, returning ABSTAIN on ambiguity."""
    selected_backend = backend or FullValueBackend()
    try:
        parameters = dict(model.named_parameters(remove_duplicate=False))
        raw: dict[str, object] = {
            "buffer": nested_named_values(dict(model.named_buffers(remove_duplicate=False))),
            "gradient": nested_named_values(
                {name: parameter.grad for name, parameter in parameters.items()}
            ),
            "model": nested_named_values(parameters),
        }
        if optimizer is not MISSING:
            raw["optimizer"] = (
                None if optimizer is None else canonicalize_optimizer(model, optimizer)
            )
        if scheduler is not MISSING:
            raw["scheduler"] = _stateful_value(scheduler)
        if scaler is not MISSING:
            raw["scaler"] = _stateful_value(scaler)
        if extras is not None:
            extra_values: dict[str, object] = {}
            for name, value in extras.items():
                state_dict = getattr(value, "state_dict", None)
                extra_values[name] = dict(state_dict()) if callable(state_dict) else value
            raw["extra"] = extra_values
        if capture_rng:
            raw["rng"] = _rng_state()
        frozen = selected_backend.freeze(raw)
        if not isinstance(frozen, FrozenMapping):
            raise TypeError("snapshot root backend result must be FrozenMapping")
        return CaptureResult(Outcome.PASS, Snapshot(step, frozen, selected_backend.name))
    except (AmbiguousStateNameError, OptimizerMappingError, UnsupportedStateError) as error:
        path = getattr(error, "path", ())
        return CaptureResult(
            Outcome.ABSTAIN,
            issue=CaptureIssue(render_path(path), str(error)),
        )
    except Exception as error:  # adapter/state_dict failures are infrastructure errors
        return CaptureResult(
            Outcome.ERROR,
            issue=CaptureIssue("$", f"{type(error).__name__}: {error}"),
        )
