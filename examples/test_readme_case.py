"""Complete PyTorch DataLoader coverage case shown in the README."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from trainparity import ExactlyOnce, Outcome
from trainparity.api import SampleCoverageResult, audit_rank_iterables


@dataclass(frozen=True)
class CoverageCase:
    """Audit stable IDs produced by one rank in one finite epoch."""

    sample_ids: tuple[int, ...]
    expected_ids: tuple[int, ...] = (0, 1, 2, 3)

    @staticmethod
    def extract(batch: list[torch.Tensor]) -> list[int]:
        return [int(value) for value in batch[0]]

    def run(self) -> SampleCoverageResult:
        dataset = TensorDataset(torch.tensor(self.sample_ids))
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        return audit_rank_iterables(
            {0: loader},
            sample_id_extractor=self.extract,
            policy=ExactlyOnce(self.expected_ids),
        )


def test_clean_loader_passes() -> None:
    assert CoverageCase((0, 1, 2, 3)).run().outcome is Outcome.PASS


def test_duplicate_loader_reports_first_observed_path() -> None:
    result = CoverageCase((0, 1, 1, 3)).run()
    assert result.outcome is Outcome.FAIL
    assert result.first_violation is not None
    assert result.first_violation.path == "coverage.same_rank_duplicate"
