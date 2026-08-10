"""Commands and tiny deterministic input for the pinned nanoGPT recipe."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np

SPLIT_STEP = 2
TOTAL_STEP = 4


def prepare(checkout: Path, _: Path) -> None:
    data_dir = checkout / "data" / "tiny_gate4_friction"
    data_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
    tokens.tofile(data_dir / "train.bin")
    tokens.tofile(data_dir / "val.bin")
    with (data_dir / "meta.pkl").open("wb") as stream:
        pickle.dump({"vocab_size": 16}, stream)


def command(
    checkout: Path, run_dir: Path, end_step: int, resume: Path | None
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        str(checkout),
        str(run_dir),
        str(end_step),
        "" if resume is None else str(resume),
    ]


def _worker(checkout: Path, run_dir: Path, end_step: int, resume: Path | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "process.pid").write_text(str(os.getpid()), encoding="utf-8")
    command_line = [
        sys.executable,
        "train.py",
        f"--out_dir={run_dir}",
        "--dataset=tiny_gate4_friction",
        f"--device={'cuda' if os.environ.get('TRAINPARITY_GATE4_DEVICE') == 'cuda' else 'cpu'}",
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
    if resume is not None:
        command_line.append("--init_from=resume")
    os.chdir(checkout)
    os.execv(sys.executable, command_line)


if __name__ == "__main__":
    if len(sys.argv) != 6 or sys.argv[1] != "worker":
        raise SystemExit("invalid nanoGPT glue arguments")
    _worker(
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        int(sys.argv[4]),
        None if not sys.argv[5] else Path(sys.argv[5]),
    )
