from __future__ import annotations

import json
import sys
from pathlib import Path

from torch.utils.data import DistributedSampler

from trainparity import ExpectedPadding
from trainparity.api import audit_rank_iterables

dataset = list(range(10))
ranks = {}
for rank in range(3):
    sampler = DistributedSampler(dataset, num_replicas=3, rank=rank, shuffle=False)
    ranks[rank] = [list(sampler)]
result = audit_rank_iterables(
    ranks,
    sample_id_extractor=lambda batch: batch,
    policy=ExpectedPadding(dataset, padding_count=2),
)
Path(sys.argv[1]).write_text(json.dumps(result.to_dict(), sort_keys=True), encoding="utf-8")
raise SystemExit(0 if result.outcome.value == "PASS" else 1)
