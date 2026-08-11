"""Explicit accumulation semantics for the pinned nanoGPT model."""

import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from trainparity import LossAccounting, TrainingState


class Case:
    equivalence = "same ordered token windows; dropout disabled; global token cross-entropy mean"

    def build(self, seed: int, device: str) -> TrainingState:
        torch.manual_seed(seed)
        config = GPTConfig(block_size=8, vocab_size=32, n_layer=1, n_head=1, n_embd=8, dropout=0.0, bias=False)
        model = GPT(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        return TrainingState(model, optimizer, scheduler)

    def batch(self, device: str) -> object:
        tokens = torch.arange(32, device=device).reshape(4, 8) % 31
        return {"tokens": tokens, "targets": (tokens + 1) % 31}

    def loss(self, state: TrainingState, batch: object) -> LossAccounting:
        assert isinstance(batch, dict)
        logits, _ = state.model(batch["tokens"])
        targets = batch["targets"]
        numerator = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
        count = int(targets.numel())
        return LossAccounting(numerator / count, numerator, count)
