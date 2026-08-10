"""Explicit nanoGPT command and checkpoint semantics; no orchestration."""

import os
import sys
from pathlib import Path
from typing import Any

import torch

from trainparity import ProcessExecutionPlan


class Case:
    name, split_step, total_step = "nanogpt", 2, 4

    def command(self, plan: ProcessExecutionPlan) -> list[str]:
        device = "cuda" if os.environ.get("TRAINPARITY_GATE4_DEVICE") == "cuda" else "cpu"
        arguments = f"--out_dir={plan.run_dir} --dataset=tiny_gate4b --device={device} --dtype=float32 --compile=False --eval_interval=1 --eval_iters=1 --log_interval=1000 --always_save_checkpoint=True --gradient_accumulation_steps=1 --batch_size=1 --block_size=8 --n_layer=1 --n_head=1 --n_embd=8 --dropout=0.0 --bias=False --learning_rate=0.001 --decay_lr=False --warmup_iters=0 --max_iters={plan.end_step} --wandb_log=False".split()
        return [sys.executable, "train.py", *arguments] + ([] if plan.resume_from is None else ["--init_from=resume"])

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "ckpt.pt"

    def observe_checkpoint(self, path: Path) -> dict[str, object]:
        value: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        return {key: value[key] for key in ("model", "optimizer", "model_args", "iter_num", "best_val_loss")}
