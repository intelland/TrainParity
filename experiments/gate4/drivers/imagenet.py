"""Run the unmodified pytorch/examples ImageNet recipe on four tiny images."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def prepare(data_root: Path) -> None:
    """Generate a deterministic two-class ImageFolder without a download."""
    from PIL import Image

    for split in ("train", "val"):
        directory = data_root / split / "only"
        directory.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 256), color=(128, 128, 128)).save(directory / "0.png")


def run(checkout: Path, data_root: Path, run_dir: Path, end_step: int, resume: Path | None) -> None:
    """Replace this process with the original upstream training script."""
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(checkout / "imagenet" / "main.py"),
        str(data_root),
        "--arch",
        "shufflenet_v2_x0_5",
        "--workers",
        "0",
        "--epochs",
        str(end_step),
        "--batch-size",
        "4",
        "--lr",
        "0.01",
        "--seed",
        "23",
        "--print-freq",
        "1000",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    os.chdir(run_dir)
    os.execv(sys.executable, command)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "prepare":
        prepare(Path(sys.argv[2]))
        return
    if len(sys.argv) not in {6, 7} or sys.argv[1] != "run":
        raise SystemExit("imagenet driver arguments are invalid")
    resume = None if len(sys.argv) == 6 else Path(sys.argv[6])
    run(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]), resume)


if __name__ == "__main__":
    main()
