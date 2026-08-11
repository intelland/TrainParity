from __future__ import annotations

import json
from pathlib import Path

from scripts import release_audit, release_smoke


def test_first_observed_extracts_difference_violation_and_phase() -> None:
    assert release_smoke._first_observed({"primary_difference": {"path": "model.weight"}}) == "model.weight"
    assert release_smoke._first_observed({"first_violation": {"path": "coverage.id"}}) == "coverage.id"
    assert release_smoke._first_observed({"first_observed_phase": "gradient"}) == "gradient"
    assert release_smoke._first_observed({}) is None


def test_repository_audit_classifies_historical_metadata(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "experiments").mkdir()
    (tmp_path / "src" / "clean.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "experiments" / "history.json").write_text(
        json.dumps({"path": "/scratch/project/checkpoint"}), encoding="utf-8"
    )
    report = release_audit._scan_repository(tmp_path)
    assert report["secret_matches"] == []
    assert report["release_facing_local_metadata"] == {}
    assert report["local_metadata_summary"][
        "accepted_or_recorded_evidence:cluster_absolute_path"
    ]["occurrences"] == 1
