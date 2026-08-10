"""Stable parameter-name canonicalization for PyTorch optimizers."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.optim import Optimizer

from trainparity.state import StatePath, nested_named_values, render_path


@dataclass(frozen=True)
class OptimizerMappingError(ValueError):
    """An optimizer state cannot be mapped to one stable parameter name."""

    path: StatePath
    detail: str

    def __str__(self) -> str:
        return f"{self.detail} at {render_path(self.path)}"


def _parameter_names(model: nn.Module) -> dict[int, tuple[str, ...]]:
    names: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        names.setdefault(id(parameter), []).append(name)
    return {key: tuple(value) for key, value in names.items()}


def canonicalize_optimizer(model: nn.Module, optimizer: Optimizer) -> dict[str, object]:
    """Replace optimizer parameter objects/IDs with unambiguous model names."""
    names_by_id = _parameter_names(model)
    ordered_parameters: list[nn.Parameter] = []
    canonical_groups: list[dict[str, object]] = []
    seen: set[int] = set()

    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise OptimizerMappingError(
                ("optimizer", "param_groups", group_index, "params"),
                "parameter group does not contain a list",
            )
        group_names: list[str] = []
        for parameter_index, parameter in enumerate(parameters):
            path = ("optimizer", "param_groups", group_index, "params", parameter_index)
            if not isinstance(parameter, nn.Parameter):
                raise OptimizerMappingError(path, "optimizer entry is not a Parameter")
            names = names_by_id.get(id(parameter), ())
            if len(names) != 1:
                detail = "parameter has no model name" if not names else f"parameter has aliases {names}"
                raise OptimizerMappingError(path, detail)
            if id(parameter) in seen:
                raise OptimizerMappingError(path, "parameter occurs more than once")
            seen.add(id(parameter))
            ordered_parameters.append(parameter)
            group_names.append(names[0])
        canonical_group = {key: value for key, value in group.items() if key != "params"}
        canonical_group["params"] = group_names
        canonical_groups.append(canonical_group)

    state_by_name: dict[str, object] = {}
    for parameter, state in optimizer.state.items():
        path = ("optimizer", "state")
        if not isinstance(parameter, nn.Parameter):
            raise OptimizerMappingError(path, "optimizer state key is not a Parameter")
        names = names_by_id.get(id(parameter), ())
        if len(names) != 1 or id(parameter) not in seen:
            raise OptimizerMappingError(path, "optimizer state key is not uniquely mapped")
        state_by_name[names[0]] = dict(state)
    for parameter in ordered_parameters:
        name = names_by_id[id(parameter)][0]
        state_by_name.setdefault(name, {})

    try:
        canonical_state = nested_named_values(state_by_name)
    except ValueError as error:
        raise OptimizerMappingError(("optimizer", "state"), str(error)) from error
    return {"param_groups": canonical_groups, "state": canonical_state}

