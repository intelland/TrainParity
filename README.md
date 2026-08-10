# TrainParity

TrainParity is an early-stage, single-process PyTorch testing library for
user-declared training equivalence. It is designed to report the first
**observed** divergence between two controlled executions; it does not infer a
root cause or diagnose arbitrary training scripts.

Gate 1 provides only the installable package skeleton and selected adapter
contract. Snapshot comparison and a fresh-process resume runner are not yet
implemented.

## Adapter contract

A resume case is an importable, zero-argument class implementing four methods:

```python
class MyCase:
    def build(self, seed: int) -> TrainingState: ...
    def train_step(self, state: TrainingState) -> None: ...
    def save(self, state: TrainingState, path: Path) -> None: ...
    def load(self, path: Path, seed: int) -> TrainingState: ...
```

The user owns construction, one logical training step, and checkpoint
semantics. TrainParity does not modify the training loop. A case is referenced
as `package.module:ClassName`, making it importable in a new Python process.

The Gate 1 example can be inspected after installation:

```bash
trainparity inspect trainparity.examples.resume_cases:CorrectResumeCase
```

This command validates only import and protocol shape. It does not claim
resume equivalence.

## Development

All runtime verification for this repository is performed on M3:

```bash
make lint
make typecheck
make test
make build
python scripts/verify_gate.py 1
```

## Runtime dependency

The sole production dependency is `torch>=2.5`. TrainParity's public state
contract contains PyTorch modules, optimizers, and schedulers, so hiding this
dependency behind an optional extra would make installation and typing
misleading. Build, Ruff, Mypy, pytest, and coverage are development-only.

## Current limitations

- Gate 1 supports only the adapter contract and import inspection.
- There is no production snapshot, comparator, or resume runner yet.
- There is no support for distributed training, Lightning, Transformers,
  services, dashboards, registries, or runtime LLM/agent integration.
- The examples are tiny CPU fixtures, not evidence of broad compatibility.

See [docs/PRODUCT_CONTRACT.md](docs/PRODUCT_CONTRACT.md) and
[docs/API_PROTOTYPES.md](docs/API_PROTOTYPES.md) for the precise boundaries.

