"""Fresh worker for one bounded accumulation execution."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from trainparity.accumulation import AccumulationExecutionPlan, UnsafeBatchSplit, split_tensor_tree
from trainparity.importing import load_accumulation_case
from trainparity.optimizer_state import canonicalize_optimizer
from trainparity.outcomes import Outcome
from trainparity.protocols import LossAccounting, TrainingState
from trainparity.serialization import encode_snapshot
from trainparity.snapshot import Snapshot, Stateful, capture_snapshot
from trainparity.state import FrozenMapping, FullValueBackend, nested_named_values


def _snapshot(value: Mapping[str, object]) -> Snapshot:
    frozen = FullValueBackend().freeze(value)
    if not isinstance(frozen, FrozenMapping):
        raise TypeError("phase snapshot root must be a mapping")
    return Snapshot(1, frozen)


def _model_state(state: TrainingState) -> dict[str, object]:
    return {
        "model": nested_named_values(dict(state.model.named_parameters(remove_duplicate=False))),
        "buffer": nested_named_values(dict(state.model.named_buffers(remove_duplicate=False))),
    }


def _gradient_state(state: TrainingState) -> dict[str, object]:
    return {
        "gradient": nested_named_values(
            {name: parameter.grad for name, parameter in state.model.named_parameters(remove_duplicate=False)}
        )
    }


def _scheduler_state(state: TrainingState) -> dict[str, object]:
    scheduler_method = cast(
        Callable[[], Mapping[str, object]] | None,
        None if state.scheduler is None else state.scheduler.state_dict,
    )
    scheduler = None if scheduler_method is None else dict(scheduler_method())
    scaler_method = cast(
        Callable[[], Mapping[str, object]] | None,
        getattr(state.scaler, "state_dict", None),
    )
    scaler = None if scaler_method is None else dict(scaler_method())
    return {"scheduler": scheduler, "scaler": scaler}


def _split(case: object, batch: object, count: int) -> tuple[object, ...]:
    explicit = getattr(case, "split_batch", None)
    values = explicit(batch, count) if callable(explicit) else split_tensor_tree(batch, count)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise UnsafeBatchSplit("explicit splitter must return an ordered sequence")
    result = tuple(values)
    if len(result) != count:
        raise UnsafeBatchSplit("explicit splitter returned the wrong microbatch count")
    return result


def _accounting(
    terms: Sequence[LossAccounting], plan: AccumulationExecutionPlan
) -> tuple[list[torch.Tensor], dict[str, object], bool]:
    explicit = [term.numerator is not None and term.denominator is not None for term in terms]
    if any(explicit) and not all(explicit):
        raise UnsafeBatchSplit("loss accounting must be supplied for every microbatch or none")
    if all(explicit) and plan.use_explicit_loss_accounting:
        denominators = [float(term.denominator) for term in terms]  # type: ignore[arg-type]
        if any(value <= 0 for value in denominators):
            raise UnsafeBatchSplit("loss denominators must be positive")
        total_denominator = sum(denominators)
        numerators = [term.numerator for term in terms]
        assert all(value is not None for value in numerators)
        effective = [value / total_denominator for value in numerators if value is not None]
        total_numerator = sum((value.detach() for value in numerators if value is not None), torch.zeros((), device=effective[0].device))
        return effective, {
            "effective_loss": total_numerator / total_denominator,
            "numerator": total_numerator,
            "denominator": total_denominator,
            "normalization_captured": True,
        }, True
    divisor = len(terms) if plan.scale_accumulated_loss else 1
    effective = [term.value / divisor for term in terms]
    total = sum((value.detach() for value in effective), torch.zeros((), device=effective[0].device))
    return effective, {
        "effective_loss": total,
        "normalization_captured": False,
    }, False


def execute(case_spec: str, plan: AccumulationExecutionPlan, device: str, seed: int) -> dict[str, Any]:
    """Execute one plan and return serialized bounded observations."""
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    case = load_accumulation_case(case_spec)
    state = case.build(seed, device)
    captured = capture_snapshot(
        state.model, step=0, optimizer=state.optimizer, scheduler=state.scheduler,
        scaler=cast(Stateful | None, state.scaler), capture_rng=True,
    )
    if captured.outcome is not Outcome.PASS or captured.snapshot is None:
        return {
            "outcome": captured.outcome.value,
            "message": "initial state capture failed",
            "pid": os.getpid(),
            "equivalence": case.equivalence,
        }
    batch = case.batch(device)
    microbatches = list(_split(case, batch, plan.microbatch_count))
    if plan.omit_final_microbatch:
        microbatches = microbatches[:-1]
    if not microbatches:
        raise UnsafeBatchSplit("plan consumed no microbatches")

    state.optimizer.zero_grad(set_to_none=True)
    terms = [case.loss(state, item) for item in microbatches]
    effective, loss_state, loss_captured = _accounting(terms, plan)
    scaler = state.scaler
    for loss in effective:
        scaled = getattr(scaler, "scale", None)
        (scaled(loss) if callable(scaled) else loss).backward()
        if plan.clip_per_microbatch and plan.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), plan.clip_grad_norm)
        if plan.optimizer_step_per_microbatch:
            step_method = getattr(scaler, "step", None)
            if callable(step_method) and not plan.amp_step_before_unscale:
                step_method(state.optimizer)
                update = getattr(scaler, "update", None)
                if callable(update):
                    update()
            else:
                state.optimizer.step()
            state.optimizer.zero_grad(set_to_none=True)
        if plan.scheduler_step_per_microbatch and state.scheduler is not None:
            state.scheduler.step()

    unscale = getattr(scaler, "unscale_", None)
    if callable(unscale) and not plan.optimizer_step_per_microbatch and not plan.amp_step_before_unscale:
        unscale(state.optimizer)
    if plan.clip_grad_norm is not None and not plan.clip_per_microbatch:
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), plan.clip_grad_norm)
    if plan.zero_grad_before_gradient_observation:
        state.optimizer.zero_grad(set_to_none=True)

    phases = {
        "loss_accounting": _snapshot({"loss_accounting": loss_state}),
        "gradient": _snapshot(_gradient_state(state)),
    }
    if not plan.optimizer_step_per_microbatch:
        step_method = getattr(scaler, "step", None)
        if callable(step_method) and not plan.amp_step_before_unscale:
            step_method(state.optimizer)
            update = getattr(scaler, "update", None)
            if callable(update):
                update()
        else:
            state.optimizer.step()
            update = getattr(scaler, "update", None)
            if callable(update) and plan.amp_step_before_unscale:
                current = getattr(scaler, "get_scale", lambda: 1.0)()
                update(new_scale=current)
    phases["optimizer_state"] = _snapshot({"optimizer": canonicalize_optimizer(state.model, state.optimizer)})
    phases["parameter_update"] = _snapshot(_model_state(state))
    if not plan.scheduler_step_per_microbatch and state.scheduler is not None:
        state.scheduler.step()
    phases["scheduler_state"] = _snapshot(_scheduler_state(state))
    return {
        "outcome": Outcome.PASS.value,
        "message": "one optimizer-update window completed",
        "pid": os.getpid(),
        "equivalence": case.equivalence,
        "initial": encode_snapshot(captured.snapshot),
        "phases": {name: encode_snapshot(value) for name, value in phases.items()},
        "loss_normalization_captured": loss_captured,
    }


def main() -> int:
    """Worker entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--result", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        raw_plan = json.loads(arguments.plan)
        if not isinstance(raw_plan, dict):
            raise ValueError("plan must be a JSON object")
        payload = execute(
            arguments.case, AccumulationExecutionPlan.from_dict(raw_plan),
            arguments.device, arguments.seed,
        )
    except UnsafeBatchSplit as error:
        payload = {"outcome": Outcome.ABSTAIN.value, "message": str(error), "pid": os.getpid()}
    except Exception as error:
        payload = {
            "outcome": Outcome.ERROR.value,
            "message": f"worker failed: {type(error).__name__}: {error}",
            "pid": os.getpid(),
        }
    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.result.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
