from __future__ import annotations

import json
import tomllib
from pathlib import Path

import torch

import trainparity
from trainparity import api
from trainparity.api import (
    MACHINE_REPORT_SCHEMA_VERSION,
    ExactlyOnce,
    SampleObservation,
    audit_sample_coverage,
)
from trainparity.quickstarts import accumulation, resume, sample_coverage
from trainparity.version import PACKAGE_VERSION

TOP_LEVEL_API = {
    "check_resume",
    "check_accumulation",
    "audit_sample_coverage",
    "ExactlyOnce",
    "AtLeastOnce",
    "NoCrossRankOverlap",
    "ExpectedPadding",
    "ExactComparison",
    "ToleranceComparison",
    "Outcome",
    "__version__",
}


def test_frozen_public_api_excludes_internal_runners_and_backends() -> None:
    assert set(trainparity.__all__) == TOP_LEVEL_API
    metadata = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert trainparity.__version__ == PACKAGE_VERSION == metadata["project"]["version"]
    assert PACKAGE_VERSION == "0.1.1.dev0"
    assert MACHINE_REPORT_SCHEMA_VERSION == 2
    for internal in (
        "AccumulationRunner",
        "ProcessResumeRunner",
        "Snapshot",
        "capture_snapshot",
        "PACKAGE_VERSION",
        "ExternalProcessEvidence",
        "SampleCoverageAuditor",
    ):
        assert internal not in trainparity.__all__
        assert internal not in api.__all__


def test_public_machine_report_has_schema_and_package_versions(tmp_path: Path) -> None:
    evidence = tmp_path / "coverage.json"
    result = audit_sample_coverage(
        [SampleObservation("sample-1", 0, 0, 0)],
        ExactlyOnce(("sample-1",)),
        evidence_path=evidence,
    )
    assert result.to_dict()["schema_version"] == MACHINE_REPORT_SCHEMA_VERSION
    assert result.to_dict()["trainparity_version"] == PACKAGE_VERSION
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["schema_version"] == MACHINE_REPORT_SCHEMA_VERSION
    assert payload["trainparity_version"] == PACKAGE_VERSION
    assert payload["traces"][0]["occurrences"][0]["worker"] is None


def test_three_quickstarts_have_clean_pass_and_intentional_fail() -> None:
    for payload in (sample_coverage.run(), accumulation.run(), resume.run()):
        assert payload["schema_version"] == MACHINE_REPORT_SCHEMA_VERSION
        assert payload["trainparity_version"] == PACKAGE_VERSION
        clean = payload["clean"]
        fault = payload["intentional_fail"]
        assert isinstance(clean, dict) and clean["outcome"] == "PASS"
        assert isinstance(fault, dict) and fault["outcome"] == "FAIL"
        assert clean["schema_version"] == MACHINE_REPORT_SCHEMA_VERSION
        assert fault["trainparity_version"] == PACKAGE_VERSION


def test_quickstart_case_semantics_are_directly_observable(tmp_path: Path) -> None:
    accumulation_case = accumulation.Case()
    state = accumulation_case.build(seed=17, device="cpu")
    batch = accumulation_case.batch("cpu")
    accounting = accumulation_case.loss(state, batch)
    assert accounting.denominator == 4
    assert torch.isfinite(accounting.value)

    continuous = tmp_path / "continuous"
    resume._train(continuous, end_step=2, resume_from=None, fault=False)
    checkpoint = continuous / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["step"] == saved["scheduler"]["last_epoch"] == 2

    interrupted = tmp_path / "interrupted"
    resume._train(interrupted, end_step=4, resume_from=checkpoint, fault=True)
    faulty = torch.load(
        interrupted / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert faulty["step"] == 4
    assert faulty["scheduler"]["last_epoch"] == 2
