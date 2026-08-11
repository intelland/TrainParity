"""Create the Gate 7I human-review ZIP with real repository paths."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from scripts.verify_gate7i import EXPECTED_BUNDLE


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/gate7i/trainparity-gate7i-human-review.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = sorted(relative for relative in EXPECTED_BUNDLE if not (root / relative).is_file())
    if missing:
        raise SystemExit(f"missing review files: {missing}")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(EXPECTED_BUNDLE):
            archive.write(root / relative, arcname=relative)
    with zipfile.ZipFile(output) as archive:
        observed = set(archive.namelist())
    if observed != EXPECTED_BUNDLE:
        raise SystemExit("review ZIP paths do not match the real-path inventory")
    print(
        json.dumps(
            {
                "status": "PASS",
                "entries": len(observed),
                "path": output.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
