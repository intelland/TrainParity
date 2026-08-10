"""Minimal hand-written final-model equality test used as Gate 4 control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def final_state_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(final_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, Sequence) and isinstance(right, Sequence) and not isinstance(left, (str, bytes)):
        return len(left) == len(right) and all(final_state_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right

