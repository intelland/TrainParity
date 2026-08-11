from __future__ import annotations

import json
import sys
from pathlib import Path

from torch.utils.data import SequentialSampler

from trainparity import ExactlyOnce, audit_rank_iterables


dataset = list(range(11))
sampler = SequentialSampler(dataset)
result = audit_rank_iterables(
    {0: [list(sampler)]},
    sample_id_extractor=lambda batch: batch,
    policy=ExactlyOnce(dataset),
)
Path(sys.argv[1]).write_text(json.dumps(result.to_dict(), sort_keys=True), encoding="utf-8")
raise SystemExit(0 if result.outcome.value == "PASS" else 1)
