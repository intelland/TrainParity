"""Audit repository text and built distributions for release-boundary leaks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Any

VERSION = "0.1.0"
TEXT_SUFFIXES = {".json", ".md", ".py", ".ps1", ".sh", ".toml", ".txt", ".yaml", ".yml"}
EXCLUDED_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "build", "dist", "__pycache__"}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[opurs]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "email_token_query": re.compile(r"[?&]email_token=[A-Za-z0-9]+"),
}
LOCAL_PATTERNS = {
    "windows_user_absolute_path": re.compile(
        r"\b[A-Za-z]:" + r"\\" + "Users" + r"\\[^\\\s\"'`]+(?:\\[^\s\"'`]*)?"
    ),
    "macos_user_absolute_path": re.compile(
        "/" + "Users/" + r"[^/\s\"'`]+(?:/[^\s\"'`]*)?"
    ),
    "linux_home_absolute_path": re.compile(
        "/" + "home/" + r"[^/\s\"'`]+(?:/[^\s\"'`]*)?"
    ),
    "scratch_absolute_path": re.compile("/" + "scratch" + r"(?:/[^\s\"'`]*)?"),
}
UNWANTED_SUFFIXES = {".ckpt", ".npy", ".npz", ".pt", ".pth", ".pyc"}
FORBIDDEN_WHEEL = ("trainparity/examples/",)
FORBIDDEN_SDIST = (
    "scripts/",
    "tests/",
    ".github/",
    "AGENTS.md",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
    )


def _scan_repository(root: Path) -> dict[str, Any]:
    secrets: list[dict[str, str]] = []
    local_metadata: dict[str, dict[str, int]] = {}
    for path in _text_files(root):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                secrets.append({"file": relative, "pattern": name})
        for name, pattern in LOCAL_PATTERNS.items():
            count = len(pattern.findall(text))
            if count:
                local_metadata.setdefault(name, {"files": 0, "occurrences": 0})
                local_metadata[name]["files"] += 1
                local_metadata[name]["occurrences"] += count
    unwanted = []
    large = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(
            part in EXCLUDED_PARTS for part in path.relative_to(root).parts
        ):
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in UNWANTED_SUFFIXES:
            unwanted.append(relative)
        if path.stat().st_size > 5 * 1024 * 1024:
            large.append({"file": relative, "bytes": path.stat().st_size})
    return {
        "secret_matches": secrets,
        "local_metadata_summary": local_metadata,
        "checkpoint_dataset_cache_files": unwanted,
        "large_files_over_5mb": large,
    }


def _audit_archives(root: Path) -> dict[str, Any]:
    wheel = root / "dist" / f"trainparity-{VERSION}-py3-none-any.whl"
    sdist = root / "dist" / f"trainparity-{VERSION}.tar.gz"
    if not wheel.is_file() or not sdist.is_file():
        raise RuntimeError("expected wheel and sdist were not built")
    with zipfile.ZipFile(wheel) as archive:
        wheel_files = sorted(archive.namelist())
        wheel_secret_matches = [
            name
            for name in wheel_files
            if any(pattern.search(archive.read(name).decode("utf-8", errors="ignore")) for pattern in SECRET_PATTERNS.values())
        ]
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_files = sorted(member.name for member in archive.getmembers() if member.isfile())
        sdist_secret_matches = []
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            text = handle.read().decode("utf-8", errors="ignore") if handle is not None else ""
            if any(pattern.search(text) for pattern in SECRET_PATTERNS.values()):
                sdist_secret_matches.append(member.name)
    sdist_relative = [name.split("/", 1)[1] for name in sdist_files if "/" in name]
    forbidden_wheel = [name for name in wheel_files if name.startswith(FORBIDDEN_WHEEL)]
    forbidden_sdist = [
        name
        for name in sdist_relative
        if any(name == prefix.rstrip("/") or name.startswith(prefix) for prefix in FORBIDDEN_SDIST)
    ]
    return {
        "wheel": {
            "file": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": _sha256(wheel),
            "files": wheel_files,
            "forbidden_files": forbidden_wheel,
            "secret_matches": wheel_secret_matches,
        },
        "sdist": {
            "file": sdist.name,
            "bytes": sdist.stat().st_size,
            "sha256": _sha256(sdist),
            "files": sdist_relative,
            "forbidden_files": forbidden_sdist,
            "secret_matches": sdist_secret_matches,
        },
    }


def run(root: Path) -> dict[str, Any]:
    """Return the deterministic audit report, raising on a release blocker."""
    repository = _scan_repository(root)
    archives = _audit_archives(root)
    blockers = []
    if repository["secret_matches"]:
        blockers.append("secret-like credential material found in repository text")
    if repository["local_metadata_summary"]:
        blockers.append("machine-local absolute path found in repository text")
    if archives["wheel"]["forbidden_files"] or archives["sdist"]["forbidden_files"]:
        blockers.append("repository-only content entered a distribution")
    if archives["wheel"]["secret_matches"] or archives["sdist"]["secret_matches"]:
        blockers.append("secret-like credential material entered a distribution")
    report = {
        "schema_version": 1,
        "trainparity_version": VERSION,
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "repository": repository,
        "distributions": archives,
        "dependency_licenses": [
            {
                "dependency": "torch>=2.7,<2.14",
                "license": "BSD-3-Clause",
                "source": "https://github.com/pytorch/pytorch/blob/v2.13.0/LICENSE",
            }
        ],
        "environment_values_recorded_by_default": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run(root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "blockers": report["blockers"]}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
