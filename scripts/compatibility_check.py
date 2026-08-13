"""Smoke-test an installed wheel on one exact Python/PyTorch combination."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

import trainparity
from trainparity import api

MODULES = (
    "trainparity.quickstarts.resume",
    "trainparity.quickstarts.accumulation",
    "trainparity.quickstarts.sample_coverage",
)
TOP_LEVEL_API = {
    "check_resume",
    "check_accumulation",
    "audit_sample_coverage",
    "ExactlyOnce",
    "AtLeastOnce",
    "NoCrossRankOverlap",
    "ExpectedPadding",
    "ExactComparison",
    "ToleranceComparison",
    "Outcome",
    "__version__",
}


def _run(module: str, cwd: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-m", module],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{module} failed: {completed.stderr[-1000:]}")
    payload: dict[str, Any] = json.loads(completed.stdout)
    if payload["clean"]["outcome"] != "PASS":
        raise RuntimeError(f"{module} clean case did not PASS")
    if payload["intentional_fail"]["outcome"] != "FAIL":
        raise RuntimeError(f"{module} intentional case did not FAIL")
    return {
        "module": module,
        "clean": "PASS",
        "intentional_fail": "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-torch", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    observed_torch = torch.__version__.split("+", 1)[0]
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(f"expected Python 3.11, observed {sys.version.split()[0]}")
    if observed_torch != arguments.expected_torch:
        raise SystemExit(
            f"expected torch {arguments.expected_torch}, observed {torch.__version__}"
        )
    package_path = Path(trainparity.__file__).resolve()
    if package_path.is_relative_to(arguments.repository.resolve()):
        raise SystemExit("TrainParity imported from the source tree, not the installed wheel")
    if set(trainparity.__all__) != TOP_LEVEL_API:
        raise SystemExit("installed top-level API does not match the 0.1 contract")
    with tempfile.TemporaryDirectory(prefix="trainparity-compat-") as directory:
        examples = [_run(module, Path(directory)) for module in MODULES]
    report = {
        "schema_version": api.MACHINE_REPORT_SCHEMA_VERSION,
        "trainparity_version": trainparity.__version__,
        "status": "PASS",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "package_path": str(package_path),
        "source_tree_imported": False,
        "examples": examples,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "torch": torch.__version__}))


if __name__ == "__main__":
    main()
