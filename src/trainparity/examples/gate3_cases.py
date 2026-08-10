"""Small deterministic Gate 3 cases and formal checkpoint fault fixtures."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn

from trainparity.protocols import StepObservation, TrainingState

_HIDDEN_COUNTER = 0


class _Scaler(Protocol):
    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...

    def step(self, optimizer: torch.optim.Optimizer) -> Any: ...

    def update(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


@dataclass
class Gate3State(TrainingState):
    """Minimal training state with a deterministic data cursor."""

    cursor: int = 0
    last_sample_ids: tuple[int, ...] | None = None


class DeterministicCase:
    """Correct reference case used for CPU and same-device GPU checks."""

    omitted_key: str | None = None

    def build(self, seed: int) -> Gate3State:
        global _HIDDEN_COUNTER
        _HIDDEN_COUNTER = 0
        random.seed(seed)
        try:
            import numpy as np
        except ImportError as error:  # pragma: no cover - declared dev dependency
            raise RuntimeError("NumPy is required by the Gate 3 reference case") from error
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device(os.environ.get("TRAINPARITY_DEVICE", "cpu"))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = nn.Linear(3, 1).to(device)
        optimizer = torch.optim.SGD(
            [
                {"params": [model.weight], "lr": 0.04, "momentum": 0.8},
                {"params": [model.bias], "lr": 0.02, "momentum": 0.8},
            ]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
        scaler = cast(
            _Scaler,
            torch.amp.GradScaler(  # type: ignore[attr-defined]
                device.type, enabled=device.type == "cuda", growth_interval=1
            ),
        )
        return Gate3State(model, optimizer, scheduler, scaler=scaler)

    def train_step(self, state: TrainingState) -> None:
        global _HIDDEN_COUNTER
        assert isinstance(state, Gate3State)
        import numpy as np

        device = next(state.model.parameters()).device
        sample_ids = (state.cursor % 11, (state.cursor + 1) % 11)
        features = torch.tensor(
            [[sample_ids[0], 1.0, -1.0], [sample_ids[1], -0.5, 2.0]],
            dtype=torch.float32,
            device=device,
        )
        stochastic = random.random() + float(np.random.random())
        stochastic_tensor = torch.rand((), device=device) + stochastic
        target = torch.tensor([[0.25], [-0.5]], device=device) + stochastic_tensor * 0.01
        state.optimizer.zero_grad(set_to_none=True)
        loss = (state.model(features) - target).square().mean()
        assert state.scaler is not None
        scaler = cast(_Scaler, state.scaler)
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.step(state.optimizer)
        scaler.update()
        assert state.scheduler is not None
        state.scheduler.step()
        state.cursor += 2
        state.last_sample_ids = sample_ids
        state.step += 1
        _HIDDEN_COUNTER += 1

    def observe(self, state: TrainingState) -> StepObservation:
        assert isinstance(state, Gate3State)
        return StepObservation(
            sample_ids=state.last_sample_ids,
            extras={"hidden_module_counter": _HIDDEN_COUNTER},
        )

    def save(self, state: TrainingState, path: Path) -> None:
        assert isinstance(state, Gate3State)
        import numpy as np

        numpy_state = cast(tuple[str, Any, int, int, float], np.random.get_state(legacy=True))
        scaler = cast(_Scaler | None, state.scaler)
        payload: dict[str, Any] = {
            "model": state.model.state_dict(),
            "gradients": {
                name: None if parameter.grad is None else parameter.grad.detach().clone()
                for name, parameter in state.model.named_parameters()
            },
            "optimizer": state.optimizer.state_dict(),
            "scheduler": None if state.scheduler is None else state.scheduler.state_dict(),  # type: ignore[no-untyped-call]
            "scaler": None if scaler is None else scaler.state_dict(),
            "step": state.step,
            "cursor": state.cursor,
            "last_sample_ids": state.last_sample_ids,
            "hidden": _HIDDEN_COUNTER,
            "python_rng": random.getstate(),
            "numpy_rng": {
                "name": numpy_state[0],
                "keys": numpy_state[1].tolist(),
                "position": numpy_state[2],
                "has_gauss": numpy_state[3],
                "cached": numpy_state[4],
            },
            "torch_cpu_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
        if self.omitted_key is not None:
            payload.pop(self.omitted_key, None)
        torch.save(payload, path)

    def load(self, path: Path, seed: int) -> Gate3State:
        global _HIDDEN_COUNTER
        import numpy as np

        device = torch.device(os.environ.get("TRAINPARITY_DEVICE", "cpu"))
        checkpoint: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
        state = self.build(seed)
        if "model" in checkpoint:
            state.model.load_state_dict(checkpoint["model"])
        if "gradients" in checkpoint:
            for name, parameter in state.model.named_parameters():
                gradient = checkpoint["gradients"][name]
                parameter.grad = None if gradient is None else gradient.to(device)
        if "optimizer" in checkpoint:
            state.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            assert state.scheduler is not None
            state.scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and checkpoint["scaler"] is not None:
            scaler = cast(_Scaler, state.scaler)
            scaler.load_state_dict(checkpoint["scaler"])
        state.step = int(checkpoint.get("step", state.step))
        state.cursor = int(checkpoint.get("cursor", state.cursor))
        raw_ids = checkpoint.get("last_sample_ids")
        state.last_sample_ids = None if raw_ids is None else tuple(int(item) for item in raw_ids)
        _HIDDEN_COUNTER = int(checkpoint.get("hidden", _HIDDEN_COUNTER))
        if "python_rng" in checkpoint:
            random.setstate(checkpoint["python_rng"])
        if "numpy_rng" in checkpoint:
            value = checkpoint["numpy_rng"]
            np.random.set_state(
                (
                    value["name"],
                    np.asarray(value["keys"], dtype=np.uint32),
                    value["position"],
                    value["has_gauss"],
                    value["cached"],
                )
            )
        if "torch_cpu_rng" in checkpoint:
            torch.set_rng_state(checkpoint["torch_cpu_rng"].cpu())
        if "torch_cuda_rng" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng"])
        self.after_load(state)
        return state

    def after_load(self, state: Gate3State) -> None:
        """Fault hook applied only after checkpoint restoration."""


class MissingModelCase(DeterministicCase):
    omitted_key = "model"


class MissingOptimizerCase(DeterministicCase):
    omitted_key = "optimizer"


class MissingSchedulerCase(DeterministicCase):
    omitted_key = "scheduler"


class MissingPythonRngCase(DeterministicCase):
    omitted_key = "python_rng"


class MissingNumpyRngCase(DeterministicCase):
    omitted_key = "numpy_rng"


class MissingTorchCpuRngCase(DeterministicCase):
    omitted_key = "torch_cpu_rng"


class MissingCudaRngCase(DeterministicCase):
    omitted_key = "torch_cuda_rng"


class MissingGradScalerCase(DeterministicCase):
    omitted_key = "scaler"


class CursorOffsetCase(DeterministicCase):
    def after_load(self, state: Gate3State) -> None:
        state.cursor += 1


class StepOffByOneCase(DeterministicCase):
    def after_load(self, state: Gate3State) -> None:
        state.step += 1


class OptimizerGroupMismatchCase(DeterministicCase):
    def after_load(self, state: Gate3State) -> None:
        first, second = state.optimizer.param_groups
        first["lr"], second["lr"] = second["lr"], first["lr"]


class ExtraSchedulerStepCase(DeterministicCase):
    def after_load(self, state: Gate3State) -> None:
        assert state.scheduler is not None
        state.scheduler.step()


class MissingHiddenGlobalCase(DeterministicCase):
    omitted_key = "hidden"


class NondeterministicCase(DeterministicCase):
    def train_step(self, state: TrainingState) -> None:
        super().train_step(state)
        with torch.no_grad():
            next(state.model.parameters()).add_(time.time_ns() % 997)


class MissingBatchIdentityCase(DeterministicCase):
    def observe(self, state: TrainingState) -> StepObservation:
        return StepObservation(extras={"hidden_module_counter": _HIDDEN_COUNTER})


class ChildExceptionCase(DeterministicCase):
    def train_step(self, state: TrainingState) -> None:
        raise RuntimeError("intentional child exception")


class CorruptCheckpointCase(DeterministicCase):
    def save(self, state: TrainingState, path: Path) -> None:
        path.write_bytes(b"not a torch checkpoint")


class MissingCheckpointCase(DeterministicCase):
    def save(self, state: TrainingState, path: Path) -> None:
        return None


class SlowCase(DeterministicCase):
    def train_step(self, state: TrainingState) -> None:
        time.sleep(2.0)
        super().train_step(state)
