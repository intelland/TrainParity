"""Fresh-clone product-surface check for two pinned Gate 4 projects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECTS = {
    "pytorch_examples_imagenet": ("https://github.com/pytorch/examples.git", "acc295dc7b90714f1bf47f06004fc19a7fe235c4"),
    "nanogpt": ("https://github.com/karpathy/nanoGPT.git", "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"),
}


def _run(command: list[str], cwd: Path, environment: dict[str, str] | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed: {command!r}\n{completed.stdout[-1000:]}\n{completed.stderr[-2000:]}")
    return completed.stdout


def _logical_loc(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#"))


def run(root: Path, wheel: Path, output: Path, device: str, dependency_path: Path | None) -> dict[str, Any]:
    rows = []
    source = Path(__file__).parent / "user_files"
    for name, (url, commit) in PROJECTS.items():
        clone = root / name
        _run(["git", "clone", "--no-checkout", url, str(clone)], root)
        _run(["git", "checkout", "--detach", commit], clone)
        user = clone / ".trainparity_user"
        site = clone / ".trainparity_site"
        user.mkdir()
        shutil.copy2(source / name / "trainparity_accumulation.py", user)
        shutil.copy2(source / "test_accumulation.py", user)
        _run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(site), str(wheel)], clone)
        report = clone / ".trainparity_report.json"
        environment = os.environ.copy()
        paths = [str(site), str(user), str(clone)]
        if dependency_path is not None:
            paths.append(str(dependency_path))
        environment["PYTHONPATH"] = os.pathsep.join(paths)
        environment["TRAINPARITY_GATE5_DEVICE"] = device
        _run([sys.executable, str(user / "test_accumulation.py"), str(report)], clone, environment)
        tracked = _run(["git", "diff", "--numstat", commit], clone).strip()
        adapter_loc = _logical_loc(user / "trainparity_accumulation.py")
        glue_loc = _logical_loc(user / "test_accumulation.py")
        rows.append({
            "project": name,
            "url": url,
            "commit": commit,
            "adapter_logical_loc": adapter_loc,
            "supporting_glue_logical_loc": glue_loc,
            "total_user_logical_loc": adapter_loc + glue_loc,
            "upstream_modified_loc": 0 if not tracked else None,
            "result": json.loads(report.read_text(encoding="utf-8")),
            "files": [str(path.relative_to(clone)) for path in sorted(user.glob("*.py"))],
            "adapter_sha256": hashlib.sha256((user / "trainparity_accumulation.py").read_bytes()).hexdigest(),
        })
    payload = {"schema_version": 1, "device": device, "projects": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dependency-path", type=Path)
    arguments = parser.parse_args()
    arguments.root.mkdir(parents=True, exist_ok=True)
    payload = run(
        arguments.root, arguments.wheel, arguments.output, arguments.device,
        arguments.dependency_path,
    )
    passed = all(row["result"]["outcome"] == "PASS" and row["total_user_logical_loc"] <= 50 and row["upstream_modified_loc"] == 0 for row in payload["projects"])
    print(json.dumps({"projects": len(payload["projects"]), "passed": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
