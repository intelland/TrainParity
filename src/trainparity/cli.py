"""Minimal Gate 1 command-line surface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from trainparity import __version__
from trainparity.importing import CaseImportError, load_case


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect an importable adapter without claiming an equivalence result."""
    parser = argparse.ArgumentParser(prog="trainparity")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("case")
    args = parser.parse_args(argv)
    try:
        case = load_case(args.case)
    except CaseImportError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {"case": args.case, "class": type(case).__name__, "protocol": "ResumeCase"},
            sort_keys=True,
        )
    )
    return 0

