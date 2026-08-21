from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_production_surface_has_no_project_or_framework_branches() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (ROOT / "src/trainparity").rglob("*.py")
    )
    for forbidden in (
        "nanogpt",
        "imagenet",
        "ignite",
        "lightning",
        "transformers",
        "deepspeed",
    ):
        assert forbidden not in production


def test_sample_coverage_remains_an_auditor_not_a_distributed_runtime() -> None:
    source = (ROOT / "src/trainparity/sample_coverage.py").read_text(
        encoding="utf-8"
    ).lower()
    for forbidden in ("init_process_group", "nccl", "subprocess"):
        assert forbidden not in source
    assert "#SBATCH" not in source


def test_process_resume_uses_the_full_value_correctness_reference() -> None:
    worker = (ROOT / "src/trainparity/process_worker.py").read_text(
        encoding="utf-8"
    )
    assert "FullValueBackend().freeze" in worker
