"""User-owned invocation of the generic TrainParity process runner."""

import sys
from pathlib import Path

from trainparity import Outcome, check_resume

report = Path(sys.argv[1]).resolve()
result = check_resume(
    "trainparity_adapter:Case",
    cwd=Path.cwd(),
    report_path=report,
    environment={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
    timeout=300,
    temporary_root=report.parent / "temporary",
)
print(result.outcome.value, result.message)
raise SystemExit(0 if result.outcome is Outcome.PASS else 1)
