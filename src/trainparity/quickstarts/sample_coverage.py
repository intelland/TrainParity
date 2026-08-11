"""Run one clean and one intentionally faulty sample-coverage audit."""

from __future__ import annotations

import json

from trainparity.api import (
    MACHINE_REPORT_SCHEMA_VERSION,
    ExactlyOnce,
    SampleObservation,
    audit_sample_coverage,
)
from trainparity.version import PACKAGE_VERSION


def run() -> dict[str, object]:
    """Return a clean PASS and an intentional duplicate-ID FAIL."""
    clean = [SampleObservation(sample_id, 0, 0, sample_id) for sample_id in range(4)]
    faulty = [*clean[:3], SampleObservation(2, 0, 0, 3)]
    return {
        "schema_version": MACHINE_REPORT_SCHEMA_VERSION,
        "trainparity_version": PACKAGE_VERSION,
        "clean": audit_sample_coverage(clean, ExactlyOnce(range(4))).to_dict(),
        "intentional_fail": audit_sample_coverage(
            faulty, ExactlyOnce(range(4))
        ).to_dict(),
    }


def main() -> int:
    """Print bounded JSON and succeed only when both expected outcomes occur."""
    payload = run()
    print(json.dumps(payload, indent=2, sort_keys=True))
    clean = payload["clean"]
    fault = payload["intentional_fail"]
    assert isinstance(clean, dict) and isinstance(fault, dict)
    return 0 if clean["outcome"] == "PASS" and fault["outcome"] == "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
