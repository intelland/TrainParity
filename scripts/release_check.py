"""Install the built wheel in a new environment and run release smoke tests."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VERSION = "0.1.0rc1"


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True, timeout=1200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-environment", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheel = root / "dist" / f"trainparity-{VERSION}-py3-none-any.whl"
    if not wheel.is_file():
        raise SystemExit(f"missing built wheel: {wheel.name}")
    temporary_parent_value = os.environ.get("TRAINPARITY_RELEASE_TEMP_ROOT")
    temporary_parent = Path(temporary_parent_value) if temporary_parent_value else None
    if temporary_parent is not None:
        temporary_parent.mkdir(parents=True, exist_ok=True)
    environment = Path(
        tempfile.mkdtemp(prefix="trainparity-release-", dir=temporary_parent)
    )
    try:
        venv = environment / "venv"
        work = environment / "outside-repository"
        work.mkdir()
        _run([sys.executable, "-m", "venv", str(venv)])
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        torch_index = os.environ.get(
            "TRAINPARITY_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu"
        )
        clean_environment = os.environ.copy()
        clean_environment.pop("PYTHONPATH", None)
        clean_environment.pop("PYTHONHOME", None)
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "torch==2.7.0",
                "--index-url",
                torch_index,
            ],
            cwd=work,
            environment=clean_environment,
        )
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ],
            cwd=work,
            environment=clean_environment,
        )
        _run(
            [
                str(python),
                str(root / "scripts" / "release_smoke.py"),
                "--repository",
                str(root),
                "--output",
                str(root / "experiments" / "gate7" / "recorded" / "wheel_smoke.json"),
            ],
            cwd=work,
            environment=clean_environment,
        )
    finally:
        if arguments.keep_environment:
            print(f"kept release environment: {environment}")
        else:
            shutil.rmtree(environment)


if __name__ == "__main__":
    main()
