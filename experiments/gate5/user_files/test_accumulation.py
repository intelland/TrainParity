"""Shared user invocation deployed unchanged into each pinned clone."""

import json
import os
import sys
from pathlib import Path

from trainparity import ToleranceComparison, check_accumulation
from trainparity.api import AccumulationExecutionPlan

report = Path(sys.argv[1])
result = check_accumulation(
    "trainparity_accumulation:Case", candidate=AccumulationExecutionPlan(2), comparison=ToleranceComparison(rtol=1e-5, atol=1e-6), device=os.environ.get("TRAINPARITY_GATE5_DEVICE", "cpu"), report_path=report, timeout=300
)
print(json.dumps(result.to_dict(), sort_keys=True))
raise SystemExit(0 if result.outcome == "PASS" else 1)
