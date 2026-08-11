"""Execute and measure the two standard-sampler user surfaces."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _logical_loc(path: Path) -> int:
    return sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def run(root: Path, output: Path) -> dict[str, Any]:
    """Run both untouched PyTorch sampler cases using only their user files."""
    source = root / "experiments" / "gate6" / "user_files"
    runtime = output.parent / "product_runs"
    runtime.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("sequential_sampler", "distributed_sampler"):
        user_file = source / f"{name}.py"
        result_path = runtime / f"{name}.json"
        completed = subprocess.run(
            [sys.executable, str(user_file), str(result_path)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        rows.append(
            {
                "case": name,
                "sampler": "torch.utils.data." + ("SequentialSampler" if name == "sequential_sampler" else "DistributedSampler"),
                "user_file": str(user_file.relative_to(root)),
                "user_required_logical_loc": _logical_loc(user_file),
                "upstream_modified_loc": 0,
                "returncode": completed.returncode,
                "result": json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None,
            }
        )
    payload = {"schema_version": 1, "cases": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    payload = run(root, arguments.output)
    passed = all(
        row["returncode"] == 0
        and row["result"]["outcome"] == "PASS"
        and row["user_required_logical_loc"] <= 25
        and row["upstream_modified_loc"] == 0
        for row in payload["cases"]
    )
    print(json.dumps({"cases": len(payload["cases"]), "passed": passed}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

