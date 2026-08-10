"""Repeat Gate 0 fixtures and persist their deterministic A/B outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.gate0.ab_prototype import evaluate_matrix

EXPECTED = {
    "missing_scheduler_state": (2, "optimizer.lr"),
    "missing_rng_state": (2, "rng.torch"),
    "mean_of_means": (0, "gradient.model.weight"),
    "sample_duplication": (3, "batch.sample_ids.0"),
}


def main() -> None:
    """Run every fixture three times and require identical first differences."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [evaluate_matrix() for _ in range(3)]
    if runs[1:] != runs[:-1]:
        raise SystemExit("fault matrix is not repeatable")
    for result in runs[0]:
        expected_step, expected_path = EXPECTED[result["case"]]
        observed = result["first_divergence"]
        if result["outcome"] != "FAIL" or observed is None:
            raise SystemExit(f"{result['case']} was not detected")
        if (observed["step"], observed["path"]) != (expected_step, expected_path):
            raise SystemExit(f"unexpected divergence for {result['case']}: {observed}")
    payload = {"schema_version": 1, "repeat_count": 3, "stable": True, "results": runs[0]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
