"""Deterministic fault fixtures used by the Gate 0 A/B prototype."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch

Snapshot: TypeAlias = dict[str, object]
Trajectory: TypeAlias = list[Snapshot]
PairFactory: TypeAlias = Callable[[bool], Trajectory]

SEED = 20260810


def _scheduler_trajectory(fault: bool) -> Trajectory:
    torch.manual_seed(SEED)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    trajectory: Trajectory = []
    for step in range(5):
        if step == 2 and fault:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=3, gamma=0.1
            )
        optimizer.zero_grad()
        loss = (model(torch.tensor([[float(step + 1)]])) - 1.0).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        trajectory.append(
            {
                "model": {"weight": model.weight.detach().clone()},
                "optimizer": {"lr": optimizer.param_groups[0]["lr"]},
                "scheduler": {"last_epoch": scheduler.last_epoch},
            }
        )
    return trajectory


def _rng_trajectory(fault: bool) -> Trajectory:
    torch.manual_seed(SEED)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    trajectory: Trajectory = []
    for step in range(4):
        if step == 2 and fault:
            torch.manual_seed(SEED)
        trajectory.append(
            {
                "model": {"weight": model.weight.detach().clone()},
                "rng": {"torch": torch.get_rng_state().clone()},
            }
        )
        optimizer.zero_grad()
        target = torch.rand(1)
        loss = (model(torch.ones(1, 1)).flatten() - target).square().mean()
        loss.backward()
        optimizer.step()
    return trajectory


def _accumulation_trajectory(fault: bool) -> Trajectory:
    torch.manual_seed(SEED)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[1.0], [1.0], [1.0], [8.0]])
    optimizer.zero_grad()
    if fault:
        losses = [(model(x[:1]) - y[:1]).square().mean(), (model(x[1:]) - y[1:]).square().mean()]
        aggregate = sum(losses) / len(losses)
    else:
        aggregate = (model(x) - y).square().mean()
    aggregate.backward()
    gradient = model.weight.grad.detach().clone()
    optimizer.step()
    return [
        {
            "gradient": {"model": {"weight": gradient}},
            "loss": {"aggregate": aggregate.detach().clone()},
            "model": {"weight": model.weight.detach().clone()},
        }
    ]


def _sampling_trajectory(fault: bool) -> Trajectory:
    torch.manual_seed(SEED)
    model = torch.nn.Embedding(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    sample_ids = [0, 1, 2, 2] if fault else [0, 1, 2, 3]
    trajectory: Trajectory = []
    for sample_id in sample_ids:
        trajectory.append(
            {
                "batch": {"sample_ids": [sample_id]},
                "model": {"weight": model.weight.detach().clone()},
            }
        )
        optimizer.zero_grad()
        loss = model(torch.tensor([sample_id])).square().mean()
        loss.backward()
        optimizer.step()
    return trajectory


FIXTURES: dict[str, PairFactory] = {
    "missing_scheduler_state": _scheduler_trajectory,
    "missing_rng_state": _rng_trajectory,
    "mean_of_means": _accumulation_trajectory,
    "sample_duplication": _sampling_trajectory,
}


def make_pair(name: str) -> tuple[Trajectory, Trajectory]:
    """Return deterministic baseline and faulty trajectories for ``name``."""
    factory = FIXTURES[name]
    return factory(False), factory(True)
