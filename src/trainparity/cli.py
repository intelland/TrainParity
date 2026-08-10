"""Command-line surface for import inspection and resume equivalence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from trainparity import __version__
from trainparity.importing import CaseImportError, load_case
from trainparity.runner import ResumeRunner


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect an importable adapter without claiming an equivalence result."""
    parser = argparse.ArgumentParser(prog="trainparity")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("case")
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("case")
    resume_parser.add_argument("--total-steps", type=int, default=4)
    resume_parser.add_argument("--split-step", type=int, default=2)
    resume_parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.command == "resume":
        result = ResumeRunner().run(
            args.case,
            total_steps=args.total_steps,
            split_step=args.split_step,
            seed=args.seed,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.outcome.value == "PASS" else 1
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
