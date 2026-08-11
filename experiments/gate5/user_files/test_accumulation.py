"""Shared user invocation deployed unchanged into each pinned clone."""

import json
import os
import sys
from pathlib import Path

from trainparity import AccumulationExecutionPlan, AccumulationRunner, ToleranceComparison

report = Path(sys.argv[1])
result = AccumulationRunner(
    comparison=ToleranceComparison(rtol=1e-5, atol=1e-6), timeout=300
).run("trainparity_accumulation:Case", candidate=AccumulationExecutionPlan(2), device=os.environ.get("TRAINPARITY_GATE5_DEVICE", "cpu"), report_path=report)
print(json.dumps(result.to_dict(), sort_keys=True))
raise SystemExit(0 if result.outcome == "PASS" else 1)
