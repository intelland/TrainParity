from __future__ import annotations

import json
from pathlib import Path

from experiments.gate5.build_report import FAULTS

ROOT = Path(__file__).resolve().parents[1]


def test_gate5_report_contract() -> None:
    report = json.loads((ROOT / "artifacts/gate_reports/gate_5.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "GO"
    assert not report["gate_6_started"]
    assert report["contract"]["equivalence_is_user_declared"]
    assert not report["contract"]["tolerance_inferred_or_tuned"]
    assert set(row["name"] for row in report["faults"]) == set(FAULTS)
    assert all(row["detected"] for row in report["faults"])
    assert report["metrics"]["peak_temporary_directory_bytes"] > 0


def test_gate5_production_has_no_project_branch() -> None:
    production = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src/trainparity").glob("*.py"))
    for name in ("nanogpt", "imagenet", "ignite", "lightning", "deepspeed", "transformers"):
        assert name not in production.lower()
