"""Run unmodified nanoGPT with a fixed nine-token memmap dataset."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np


def prepare(checkout: Path) -> None:
    data_dir = checkout / "data" / "tiny_gate4"
    data_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
    tokens.tofile(data_dir / "train.bin")
    tokens.tofile(data_dir / "val.bin")
    with (data_dir / "meta.pkl").open("wb") as stream:
        pickle.dump({"vocab_size": 16}, stream)


def run(checkout: Path, run_dir: Path, end_step: int, resume: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if os.environ.get("TRAINPARITY_GATE4_DEVICE") == "cuda" else "cpu"
    command = [
        sys.executable,
        "train.py",
        f"--out_dir={run_dir}",
        "--dataset=tiny_gate4",
        f"--device={device}",
        "--dtype=float32",
        "--compile=False",
        "--eval_interval=1",
        "--eval_iters=1",
        "--log_interval=1000",
        "--always_save_checkpoint=True",
        "--gradient_accumulation_steps=1",
        "--batch_size=1",
        "--block_size=8",
        "--n_layer=1",
        "--n_head=1",
        "--n_embd=8",
        "--dropout=0.0",
        "--bias=False",
        "--learning_rate=0.001",
        "--decay_lr=False",
        "--warmup_iters=0",
        f"--max_iters={end_step}",
        "--wandb_log=False",
    ]
    if resume:
        command.append("--init_from=resume")
    os.chdir(checkout)
    os.execv(sys.executable, command)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "prepare":
        prepare(Path(sys.argv[2]))
        return
    if len(sys.argv) not in {5, 6} or sys.argv[1] != "run":
        raise SystemExit("nanoGPT driver arguments are invalid")
    run(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), len(sys.argv) == 6)


if __name__ == "__main__":
    main()

