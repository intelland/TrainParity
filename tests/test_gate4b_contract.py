from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ("pytorch_examples_imagenet", "nanogpt", "ignite_mnist_engine")


def _logical_lines(path: Path) -> int:
    return sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def test_gate4b_user_surface_meets_loc_contract() -> None:
    root = ROOT / "experiments/gate4b/user_files"
    support = root / "test_resume.py"
    support_loc = _logical_lines(support)
    assert support_loc <= 20
    totals = []
    for project in PROJECTS:
        adapter = root / project / "trainparity_adapter.py"
        ast.parse(adapter.read_text(encoding="utf-8"))
        adapter_loc = _logical_lines(adapter)
        assert adapter_loc <= 30
        assert adapter_loc + support_loc <= 50
        totals.append(adapter_loc + support_loc)
    assert sorted(totals)[1] <= 40


def test_user_files_do_not_import_gate_experiment_helpers() -> None:
    for path in (ROOT / "experiments/gate4b/user_files").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "experiments.gate" not in source
        assert "ProcessResumeRunner" in source or "ProcessExecutionPlan" in source


def test_production_surface_has_no_framework_or_project_branches() -> None:
    forbidden = ("nanogpt", "ignite", "imagenet", "lightning", "transformers", "deepspeed")
    production = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (ROOT / "src/trainparity").glob("*.py")
    )
    assert all(name not in production for name in forbidden)


def test_full_value_backend_remains_process_correctness_reference() -> None:
    worker = (ROOT / "src/trainparity/process_worker.py").read_text(encoding="utf-8")
    assert "FullValueBackend().freeze" in worker


def test_glue_decomposition_precedes_and_quantifies_production_move() -> None:
    document = (ROOT / "docs/GATE4_GLUE_DECOMPOSITION.md").read_text(encoding="utf-8")
    for category in (
        "generic TrainParity orchestration",
        "generic resume-testing infrastructure",
        "project-specific training semantics",
        "Gate-4-only benchmarking/fault/measurement code",
    ):
        assert category in document
    assert "646" in document and "812" in document


def test_closer_handwritten_baseline_remains_116_lines() -> None:
    path = ROOT / "experiments/gate4/friction/handwritten_fresh_resume.py"
    assert _logical_lines(path) == 116
