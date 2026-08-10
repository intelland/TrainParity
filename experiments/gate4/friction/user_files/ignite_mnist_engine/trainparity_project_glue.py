"""Commands and deterministic loader for the pinned Ignite engine recipe."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.data import DataLoader, TensorDataset

SPLIT_STEP = 2
TOTAL_STEP = 4


def prepare(_: Path, __: Path) -> None:
    return


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


def _load_example(checkout: Path) -> ModuleType:
    path = checkout / "examples" / "mnist" / "mnist_save_resume_engine.py"
    spec = importlib.util.spec_from_file_location("friction_ignite_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Ignite example: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_loaders(
    _: int, __: int
) -> tuple[DataLoader[tuple[torch.Tensor, ...]], DataLoader[tuple[torch.Tensor, ...]]]:
    images = torch.linspace(-1.0, 1.0, 4 * 28 * 28).reshape(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    return loader, loader


def _worker(checkout: Path, run_dir: Path, end_step: int, resume: Path | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "process.pid").write_text(str(os.getpid()), encoding="utf-8")
    module = _load_example(checkout)
    module.get_data_loaders = _tiny_loaders
    module.run(4, 4, end_step, 0.01, 0.5, 1000, str(run_dir), 1, resume, deterministic=True)
    candidates = sorted(
        (path for path in run_dir.glob("checkpoint_*.pt") if path.name != "checkpoint.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if not candidates:
        raise RuntimeError("Ignite example did not create an upstream checkpoint")
    shutil.copy2(candidates[-1], run_dir / "checkpoint.pt")
    torch.save(torch.get_rng_state(), run_dir / "rng_state.pt")


if __name__ == "__main__":
    if len(sys.argv) != 6 or sys.argv[1] != "worker":
        raise SystemExit("invalid Ignite glue arguments")
    _worker(
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        int(sys.argv[4]),
        None if not sys.argv[5] else Path(sys.argv[5]),
    )
