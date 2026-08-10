# Gate 1 adapter API prototypes

## Decision

TrainParity selects an importable zero-argument class checked structurally
against `ResumeCase`. A user supplies `module:ClassName`; the class owns
deterministic construction, one logical step, save, and load.

The selected adapter neither subclasses a TrainParity base class nor changes
the training loop. Structural typing keeps user code independent while an
explicit import string works in a fresh Python process.

## Prototype A: class plus protocol (selected)

```python
class ResumeCase(Protocol):
    def build(self, seed: int) -> TrainingState: ...
    def train_step(self, state: TrainingState) -> None: ...
    def save(self, state: TrainingState, path: Path) -> None: ...
    def load(self, path: Path, seed: int) -> TrainingState: ...
```

Advantages:

- one import specification identifies construction and all behavior;
- type checkers show the four required operations together;
- user state stays explicit and ordinary Python import semantics are enough;
- no serialization of closures or `cloudpickle` dependency is needed.

Cost: the case must have a zero-argument constructor, and users must move
locally defined test helpers to an importable module. TrainParity intentionally
does not promise arbitrary-script support.

## Prototype B: factory plus callbacks (not selected)

```python
@dataclass(frozen=True)
class ResumeCallbacks:
    build: Callable[[int], TrainingState]
    train_step: Callable[[TrainingState], None]
    save: Callable[[TrainingState, Path], None]
    load: Callable[[Path, int], TrainingState]
```

This shape also works without `cloudpickle` when its factory and callbacks are
module-level. It was not selected because it adds a factory invocation and four
field-to-function wiring points without reducing the required user behavior.
Closures would appear convenient but weaken process importability, so they are
not accepted as justification for this API.

## Recorded evaluation

`experiments/gate1/run_adapter_evaluation.py` measures the selected adapter's
logical lines, imports it from an unrelated working directory in a new Python
process, and directly probes the correct and missing-scheduler examples. The
probe is evidence about the API only; it is deliberately not a reusable resume
runner or comparator, which remain Gate 3 work.

The first mismatching state key is reported as a **first observed divergence**,
not a root cause.

