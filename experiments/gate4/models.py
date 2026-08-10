"""Experiment-only contract for external command-oriented project adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ExternalProjectAdapter(Protocol):
    """Describe one pinned upstream recipe without becoming a production API."""

    name: str
    repository: str
    commit: str
    license_id: str
    structure: str
    fault_name: str
    split_step: int
    total_step: int

    def prepare_command(self, checkout: Path, data_root: Path) -> list[str]: ...

    def run_command(
        self,
        checkout: Path,
        data_root: Path,
        run_dir: Path,
        end_step: int,
        resume_from: Path | None,
    ) -> list[str]: ...

    def checkpoint_path(self, run_dir: Path) -> Path: ...

    def normalize_checkpoint(self, path: Path) -> object: ...

    def inject_fault(self, path: Path) -> None: ...

    def handwritten_state(self, path: Path) -> object: ...

