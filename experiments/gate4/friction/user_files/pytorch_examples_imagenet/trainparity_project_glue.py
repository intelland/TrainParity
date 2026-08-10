"""Commands and tiny deterministic input for the pinned ImageNet recipe."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SPLIT_STEP = 2
TOTAL_STEP = 3


def prepare(_: Path, data_root: Path) -> None:
    from PIL import Image

    for split in ("train", "val"):
        directory = data_root / split / "only"
        directory.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 256), color=(128, 128, 128)).save(directory / "0.png")


def command(
    checkout: Path, run_dir: Path, end_step: int, resume: Path | None
) -> list[str]:
    command_line = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        str(checkout),
        str(run_dir),
        str(end_step),
        "" if resume is None else str(resume),
    ]
    return command_line


def _worker(checkout: Path, run_dir: Path, end_step: int, resume: Path | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "process.pid").write_text(str(os.getpid()), encoding="utf-8")
    data_root = run_dir.parent / "data"
    command_line = [
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
        command_line.extend(("--resume", str(resume)))
    os.chdir(run_dir)
    os.execv(sys.executable, command_line)


if __name__ == "__main__":
    if len(sys.argv) != 6 or sys.argv[1] != "worker":
        raise SystemExit("invalid ImageNet glue arguments")
    _worker(
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        int(sys.argv[4]),
        None if not sys.argv[5] else Path(sys.argv[5]),
    )
