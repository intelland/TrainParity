"""Small, auditable Gate 4 adapters for pinned external projects."""

from experiments.gate4.adapters.ignite_mnist import IgniteMnistAdapter
from experiments.gate4.adapters.nanogpt import NanoGptAdapter
from experiments.gate4.adapters.pytorch_examples_imagenet import ImageNetAdapter

__all__ = ["IgniteMnistAdapter", "ImageNetAdapter", "NanoGptAdapter"]

