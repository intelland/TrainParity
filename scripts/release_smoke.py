"""Smoke-test an installed TrainParity wheel from outside its source tree."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
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


def _run_example(module: str, working_directory: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-m", module],
        cwd=working_directory,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{module} exited {completed.returncode}: {completed.stderr[-1000:]}"
        )
    payload: dict[str, Any] = json.loads(completed.stdout)
    if payload["clean"]["outcome"] != "PASS":
        raise RuntimeError(f"{module} clean fixture did not PASS")
    if payload["intentional_fail"]["outcome"] != "FAIL":
        raise RuntimeError(f"{module} intentional fixture did not FAIL")
    return {
        "module": module,
        "clean": "PASS",
        "intentional_fail": "FAIL",
        "first_observed": _first_observed(payload["intentional_fail"]),
        "schema_version": payload["schema_version"],
        "trainparity_version": payload["trainparity_version"],
    }


def _first_observed(result: dict[str, Any]) -> str | None:
    primary = result.get("primary_difference")
    if isinstance(primary, dict) and isinstance(primary.get("path"), str):
        return primary["path"]
    violation = result.get("first_violation")
    if isinstance(violation, dict) and isinstance(violation.get("path"), str):
        return violation["path"]
    phase = result.get("first_observed_phase")
    return phase if isinstance(phase, str) else None


def run(working_directory: Path, repository: Path | None) -> dict[str, Any]:
    """Run public imports and all installed examples."""
    package_path = Path(trainparity.__file__).resolve()
    if repository is not None and package_path.is_relative_to(repository.resolve()):
        raise RuntimeError("TrainParity imported from the repository, not the wheel environment")
    imported = importlib.import_module("trainparity.api")
    if imported is not api:
        raise RuntimeError("public API import identity changed")
    if set(trainparity.__all__) != TOP_LEVEL_API:
        raise RuntimeError("top-level public exports do not match the release contract")
    required = {
        "check_resume",
        "check_accumulation",
        "audit_sample_coverage",
        "ExactlyOnce",
        "ToleranceComparison",
    }
    if not required <= set(api.__all__):
        raise RuntimeError(f"missing public names: {sorted(required - set(api.__all__))}")
    results = [_run_example(module, working_directory) for module in MODULES]
    return {
        "schema_version": api.MACHINE_REPORT_SCHEMA_VERSION,
        "trainparity_version": trainparity.__version__,
        "status": "PASS",
        "installed_distribution": f"trainparity-{trainparity.__version__}",
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "top_level_api_names": sorted(trainparity.__all__),
        "advanced_api_names": sorted(api.__all__),
        "examples": results,
        "source_tree_imported": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report = run(Path.cwd(), arguments.repository)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "examples": len(report["examples"])}))


if __name__ == "__main__":
    main()
