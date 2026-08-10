"""Independent tiny programs observed by TrainCheck during Gate 0."""

from __future__ import annotations

import json

import torch

SEED = 20260810


class CompetitorFixture:
    """Run one deterministic clean or faulty training-semantics example."""

    def __init__(self, case: str, fault: bool) -> None:
        torch.manual_seed(SEED)
        self.case = case
        self.fault = fault
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
        )

    def _step(self, x: torch.Tensor, target: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss = (self.model(x) - target).square().mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def run(self) -> None:
        sample_ids: list[int] = []
        losses: list[float] = []
        if self.case == "missing_scheduler_state":
            for step in range(5):
                if step == 2 and self.fault:
                    self.scheduler = torch.optim.lr_scheduler.StepLR(
                        self.optimizer, step_size=3, gamma=0.1
                    )
                losses.append(self._step(torch.tensor([[float(step + 1)]]), torch.ones(1, 1)))
                self.scheduler.step()
        elif self.case == "missing_rng_state":
            for step in range(4):
                if step == 2 and self.fault:
                    torch.manual_seed(SEED)
                losses.append(self._step(torch.ones(1, 1), torch.rand(1, 1)))
        elif self.case == "mean_of_means":
            x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
            y = torch.tensor([[1.0], [1.0], [1.0], [8.0]])
            self.optimizer.zero_grad()
            if self.fault:
                loss = ((self.model(x[:1]) - y[:1]).square().mean() + (self.model(x[1:]) - y[1:]).square().mean()) / 2
            else:
                loss = (self.model(x) - y).square().mean()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach()))
        elif self.case == "sample_duplication":
            sample_ids = [0, 1, 2, 2] if self.fault else [0, 1, 2, 3]
            for sample_id in sample_ids:
                x = torch.tensor([[float(sample_id + 1)]])
                losses.append(self._step(x, x * 0.5))
        else:
            raise ValueError(f"unknown case: {self.case}")
        print(json.dumps({"case": self.case, "fault": self.fault, "losses": losses, "sample_ids": sample_ids, "weight": self.model.weight.detach().flatten().tolist()}, sort_keys=True))
