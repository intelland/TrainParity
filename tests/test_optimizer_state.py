from __future__ import annotations

import torch
from torch import nn

from trainparity import ExactComparison, Outcome
from trainparity.snapshot import Snapshot, capture_snapshot
from trainparity.state import FrozenMapping, FrozenValue


def require_snapshot(model: nn.Module, optimizer: torch.optim.Optimizer) -> Snapshot:
    result = capture_snapshot(model, optimizer=optimizer, capture_rng=False)
    assert result.outcome is Outcome.PASS, result.issue
    assert result.snapshot is not None
    return result.snapshot


def paths(value: FrozenValue, prefix: str = "") -> set[str]:
    if not isinstance(value, FrozenMapping):
        return {prefix}
    result: set[str] = set()
    for key, child in value.entries:
        result.update(paths(child, f"{prefix}.{key}" if prefix else key))
    return result


def test_sgd_momentum_uses_parameter_name_not_memory_id() -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    observed = paths(require_snapshot(model, optimizer).state)
    assert "optimizer.state.weight.momentum_buffer" in observed
    assert "optimizer.state.bias.momentum_buffer" in observed
    assert not any(str(id(parameter)) in path for parameter in model.parameters() for path in observed)


def test_adam_states_use_stable_parameter_names() -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    observed = paths(require_snapshot(model, optimizer).state)
    assert "optimizer.state.weight.exp_avg" in observed
    assert "optimizer.state.weight.exp_avg_sq" in observed
    assert "optimizer.state.weight.step" in observed


def test_parameter_group_order_is_observable() -> None:
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
    left_optimizer = torch.optim.SGD(
        [{"params": model[0].parameters(), "lr": 0.1}, {"params": model[1].parameters(), "lr": 0.2}]
    )
    right_optimizer = torch.optim.SGD(
        [{"params": model[0].parameters(), "lr": 0.2}, {"params": model[1].parameters(), "lr": 0.1}]
    )
    result = ExactComparison().compare(
        require_snapshot(model, left_optimizer), require_snapshot(model, right_optimizer)
    )
    assert result.outcome is Outcome.FAIL
    assert result.first_difference is not None
    assert result.first_difference.path == "optimizer.param_groups[0].lr"


def test_alias_unowned_and_duplicate_parameters_abstain() -> None:
    class AliasModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.alias = self.weight

    alias_model = AliasModel()
    alias_result = capture_snapshot(
        alias_model, optimizer=torch.optim.SGD(alias_model.parameters(), lr=0.1), capture_rng=False
    )
    assert alias_result.outcome is Outcome.ABSTAIN
    assert alias_result.issue is not None and "aliases" in alias_result.issue.detail

    model = nn.Linear(1, 1)
    foreign = nn.Parameter(torch.ones(1))
    unowned = capture_snapshot(
        model, optimizer=torch.optim.SGD([foreign], lr=0.1), capture_rng=False
    )
    assert unowned.outcome is Outcome.ABSTAIN
    assert unowned.issue is not None and "no model name" in unowned.issue.detail

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.param_groups[0]["params"].append(model.weight)
    duplicate = capture_snapshot(model, optimizer=optimizer, capture_rng=False)
    assert duplicate.outcome is Outcome.ABSTAIN
    assert duplicate.issue is not None and "more than once" in duplicate.issue.detail
