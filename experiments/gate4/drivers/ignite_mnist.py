"""Run the unmodified Ignite Engine example with a tiny deterministic loader."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.data import DataLoader, TensorDataset


def _load_example(checkout: Path) -> ModuleType:
    path = checkout / "examples" / "mnist" / "mnist_save_resume_engine.py"
    spec = importlib.util.spec_from_file_location("gate4_ignite_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Ignite example: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_loaders(_: int, __: int) -> tuple[DataLoader[tuple[torch.Tensor, ...]], DataLoader[tuple[torch.Tensor, ...]]]:
    images = torch.linspace(-1.0, 1.0, 4 * 28 * 28).reshape(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    return loader, loader


def run(checkout: Path, run_dir: Path, end_step: int, resume: Path | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "prepare":
        return
    if len(sys.argv) not in {5, 6} or sys.argv[1] != "run":
        raise SystemExit("Ignite driver arguments are invalid")
    resume = None if len(sys.argv) == 5 else Path(sys.argv[5])
    run(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), resume)


if __name__ == "__main__":
    main()
