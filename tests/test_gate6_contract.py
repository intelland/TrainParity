from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gate6_report_contract() -> None:
    report = json.loads(
        (ROOT / "artifacts/gate_reports/gate_6.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS"
    assert report["module_decision"] == "INCLUDE_MODULE"
    assert not report["gate_7_started"]
    assert report["metrics"]["device"] == "cpu"
    assert not report["metrics"]["gpu_work_added"]
    assert len(report["counter_baseline"]["structural_benefits"]) >= 3


def test_gate6_production_is_auditor_not_distributed_runtime() -> None:
    production = (ROOT / "src/trainparity/sample_coverage.py").read_text(
        encoding="utf-8"
    ).lower()
    for forbidden in (
        "init_process_group",
        "nccl",
        "slurm",
        "subprocess",
        "checkpoint",
        "nanogpt",
        "imagenet",
    ):
        assert forbidden not in production


def test_gate6_product_surface_is_bounded() -> None:
    product = json.loads(
        (ROOT / "experiments/gate6/recorded/product_surface.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(product["cases"]) == 2
    assert all(case["user_required_logical_loc"] <= 25 for case in product["cases"])
    assert all(case["upstream_modified_loc"] == 0 for case in product["cases"])
    assert all(case["result"]["outcome"] == "PASS" for case in product["cases"])
