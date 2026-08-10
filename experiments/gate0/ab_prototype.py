"""Throwaway exact A/B comparator proving first-divergence output."""

from __future__ import annotations

from typing import Any

import torch

from experiments.gate0.fault_fixtures import FIXTURES, make_pair


def _summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().flatten()
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": flat[:4].tolist(),
        }
    return value


def _difference(left: Any, right: Any, path: str = "") -> dict[str, Any] | None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                return {"path": child, "baseline": "<missing>" if key not in left else _summary(left[key]), "candidate": "<missing>" if key not in right else _summary(right[key])}
            found = _difference(left[key], right[key], child)
            if found:
                return found
        return None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return {"path": f"{path}.length", "baseline": len(left), "candidate": len(right)}
        for index, (a_item, b_item) in enumerate(zip(left, right, strict=True)):
            found = _difference(a_item, b_item, f"{path}.{index}")
            if found:
                return found
        return None
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right):
            return None
        result = {"path": path, "baseline": _summary(left), "candidate": _summary(right)}
        if left.shape == right.shape and left.is_floating_point() and right.is_floating_point():
            result["max_abs_error"] = float((left - right).abs().max()) if left.numel() else 0.0
        return result
    if type(left) is type(right) and left == right:
        return None
    return {"path": path, "baseline": _summary(left), "candidate": _summary(right)}


def first_divergence(baseline: list[dict[str, object]], candidate: list[dict[str, object]]) -> dict[str, Any] | None:
    """Return the first observed exact difference between two trajectories."""
    for step, (left, right) in enumerate(zip(baseline, candidate, strict=False)):
        found = _difference(left, right)
        if found:
            return {"step": step, **found}
    if len(baseline) != len(candidate):
        return {"step": min(len(baseline), len(candidate)), "path": "trajectory.length", "baseline": len(baseline), "candidate": len(candidate)}
    return None


def evaluate_matrix() -> list[dict[str, Any]]:
    """Evaluate every Gate 0 fixture with the throwaway comparator."""
    results = []
    for name in FIXTURES:
        baseline, candidate = make_pair(name)
        difference = first_divergence(baseline, candidate)
        results.append({"case": name, "outcome": "FAIL" if difference else "PASS", "first_divergence": difference})
    return results


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate_matrix(), indent=2, sort_keys=True))
