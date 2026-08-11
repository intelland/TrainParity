# Gate 6 sample-coverage contract

Gate 6 audits a finite set of stable sample-ID observations. It does not run
training, launch ranks or workers, initialize DDP/NCCL, or infer a dataset
contract. A user either supplies `SampleObservation` records or rank-labelled
finite iterables plus one `sample_id_extractor` that returns ordered `int` or
`str` IDs. Sample contents, masks, labels, and weights are never logged.

A stable sample ID is semantically unique within the declared expected
universe: two distinct semantic samples must not share an ID. TrainParity
validates the resulting ID trajectory, not the sample contents or the truth of
the extractor. Unavailable worker provenance is represented explicitly as
`None` / JSON `null`; it is never encoded as worker 0.

## Explicit policies

- `ExactlyOnce(expected_ids)` requires every member of an explicit finite
  universe once and rejects missing, unexpected, same-rank duplicate, and
  cross-rank-overlap observations.
- `AtLeastOnce(expected_ids)` requires every member of an explicit finite
  universe, permits repetitions, and still reports their provenance.
- `NoCrossRankOverlap()` requires no ID to appear on more than one rank.
  Same-rank duplication is separately reported but is not a violation of this
  policy.
- `ExpectedPadding(expected_ids, padding_count)` requires a finite universe,
  no missing or unexpected IDs, and exactly the declared number of repeated
  observations. It reports every repeated ID, the ranks involved, and actual
  versus declared padding counts; repetition is never silently ignored.

`None` means an unknown universe and is distinct from an empty finite universe.
`ExactlyOnce(None)`, `AtLeastOnce(None)`, and `ExpectedPadding(None, ...)`
return `ABSTAIN` before consuming the iterable. This is the only honest result
for an unbounded or unknown-universe stream. A finite `IterableDataset` can be
checked when the user supplies its reliable expected universe. Each invocation
is one declared coverage window; multi-epoch users audit each epoch separately.

## Evidence and first violation

The result is one of `PASS`, `FAIL`, `ABSTAIN`, or `ERROR`. It includes summary
counts, a configurable bounded number of anomaly examples, and a deterministic
first observed policy violation. That is an observation, not a root-cause
claim. When `evidence_path` is supplied, TrainParity additionally writes every
missing/unexpected/repeated ID, distinct same-rank and cross-rank conditions,
and the complete rank/worker/epoch/position trajectory of every observed ID.

The input order is the observation order. Missing IDs and padding-count
mismatches are terminal observations because they can only be established once
the finite window ends. Resume-cursor coverage requires no checkpoint feature:
the user combines pre- and post-resume observations for the same declared
window, allowing duplicate or missing cursor behavior to be audited directly.

## Deliberate limits

TrainParity does not establish that an ID extractor is stable or that the
declared universe is correct. Worker provenance is optional because ordinary
parent-process `DataLoader` iteration does not expose the producing worker;
users with observable worker metadata may supply it in `SampleObservation`.
The module does not claim general epoch/shuffle equivalence, data-order
equivalence, sample-content equivalence, infinite-stream exactly-once behavior,
or distributed training correctness. One audit proves only the declared finite
observation window.
