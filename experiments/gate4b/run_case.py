"""Gate-4B-only fault launcher over the generic production process runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from trainparity import Outcome, ProcessResumeRunner


def _fault(project: str, path: Path) -> None:
    weights_only = project != "ignite_mnist_engine"
    value: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=weights_only)
    if project == "pytorch_examples_imagenet":
        value["scheduler"]["last_epoch"] -= 1
    elif project == "nanogpt":
        value["iter_num"] -= 1
    elif project == "ignite_mnist_engine":
        value["lr_scheduler"]["last_epoch"] -= 1
    else:
        raise ValueError(f"unknown Gate 4B project: {project}")
    torch.save(value, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path, required=True)
    args = parser.parse_args()
    result = ProcessResumeRunner(timeout=300, temporary_root=args.temporary_root).run(
        "trainparity_adapter:Case",
        cwd=Path.cwd(),
        report_path=args.report,
        environment={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        staged_checkpoint_hook=lambda path: _fault(args.project, path),
    )
    print(result.outcome.value, result.message)
    raise SystemExit(0 if result.outcome is Outcome.FAIL else 1)


if __name__ == "__main__":
    main()
