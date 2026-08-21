from __future__ import annotations

import ast
import inspect
import statistics
from pathlib import Path

from experiments.gate4.adapters import (
    IgniteMnistAdapter,
    ImageNetAdapter,
    NanoGptAdapter,
)
from experiments.gate4.handwritten import final_state_equal
from experiments.gate4.run_matrix import DRIVER_FILES, _adapter_path, _logical_lines

ADAPTERS = (ImageNetAdapter(), NanoGptAdapter(), IgniteMnistAdapter())


def test_gate4_contract_helpers_do_not_require_unix_resource_at_import_time() -> None:
    module_path = Path(__file__).resolve().parents[1] / "experiments/gate4/run_matrix.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    top_level_resource_imports = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and any(alias.name == "resource" for alias in node.names)
        )
        or (isinstance(node, ast.ImportFrom) and node.module == "resource")
    ]
    assert top_level_resource_imports == []


def test_gate4_uses_three_exact_external_commits_and_declared_licenses() -> None:
    assert {adapter.name for adapter in ADAPTERS} == {
        "pytorch_examples_imagenet",
        "nanogpt",
        "ignite_mnist_engine",
    }
    assert {adapter.structure for adapter in ADAPTERS} == {
        "conventional image classifier",
        "small language model",
        "trainer engine with scheduler state",
    }
    assert all(len(adapter.commit) == 40 for adapter in ADAPTERS)
    assert {adapter.license_id for adapter in ADAPTERS} == {"BSD-3-Clause", "MIT"}
    assert all(adapter.repository.startswith("https://github.com/") for adapter in ADAPTERS)


def test_gate4_adapter_logical_loc_median_meets_threshold() -> None:
    logical = [_logical_lines(_adapter_path(adapter), marked=True) for adapter in ADAPTERS]
    assert statistics.median(logical) <= 30


def test_gate4_normalizers_are_not_conditioned_on_the_injected_fault() -> None:
    for adapter in ADAPTERS:
        source = inspect.getsource(adapter.normalize_checkpoint)
        assert "fault" not in source
        assert adapter.fault_name not in source


def test_gate4_drivers_retain_original_upstream_checkpoint_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {
        name: (root / relative).read_text(encoding="utf-8") for name, relative in DRIVER_FILES.items()
    }
    assert 'checkout / "imagenet" / "main.py"' in sources["pytorch_examples_imagenet"]
    assert '"train.py"' in sources["nanogpt"]
    assert "os.chdir(checkout)" in sources["nanogpt"]
    assert "Checkpoint.load_objects" not in sources["ignite_mnist_engine"]
    assert "module.run(" in sources["ignite_mnist_engine"]


def test_minimal_handwritten_comparator_only_reports_final_equality() -> None:
    assert final_state_equal({"weight": [1, 2]}, {"weight": [1, 2]}) is True
    assert final_state_equal({"weight": [1, 2]}, {"weight": [1, 3]}) is False
