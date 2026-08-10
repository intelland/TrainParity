# Gate 4 Glue Decomposition (pre-production baseline)

This document freezes the Gate 4 friction implementation at local commit
`674b957` before any Gate 4B production code is changed. Logical LOC uses the
same rule as the accepted friction audit: non-blank, non-comment source lines.
The three fresh-clone integrations total 266, 275, and 271 user-required logical
lines. This decomposition does not reclassify code merely to make a threshold.

## Categories

- **Generic TrainParity orchestration** belongs in the production library. It
  plans the two continuous controls and interrupted candidate, stages the
  checkpoint, captures immutable snapshots, applies exact comparison, and
  produces four-state results.
- **Generic resume-testing infrastructure** is also framework-neutral, but is
  concerned with child launch, real process boundaries, timeout, environment,
  temporary paths, IPC, and deterministic file publication.
- **Project-specific training semantics** must remain explicit user code: the
  original command/checkpoint contract and the state selected from that
  checkpoint.
- **Gate-4-only benchmarking/fault/measurement code** creates tiny inputs, changes
  faults, times processes, reads RSS, and produces acceptance evidence. It is
  not user integration and must not move into the production API.

## Shared clean-resume file: deployed unchanged for every project

Source: `experiments/gate4/friction/user_files/trainparity_clean_resume.py`
(191 logical LOC, deployed three times).

| Physical lines / function | Logical LOC | Classification | Gate 4B disposition |
|---|---:|---|---|
| 1-26, module/imports | 20 | mixed generic orchestration/infrastructure | production modules own the imports and types |
| 27-46, `_stable_keys` | 18 | generic TrainParity orchestration | production checkpoint snapshot worker |
| 47-53, `_peak_rss` | 5 | Gate-4-only measurement | audit harness only |
| 54-96, `_execute` | 41 | generic resume-testing infrastructure, with timing/RSS measurement mixed in | production owns launch, timeout, logs, PID and environment; audit owns `/usr/bin/time` parsing |
| 97-105, `_capture` | 7 | generic TrainParity orchestration | production snapshot worker using `FullValueBackend` |
| 106-113, `_serialize` | 6 | generic orchestration/infrastructure | production atomic snapshot IPC |
| 114-117, `_tree_size` | 2 | Gate-4-only measurement | audit harness only |
| 118-122, setup timing | 5 | Gate-4-only benchmark/measurement | audit fixture setup before the production call |
| 124-138, execution plan and staging | 14 | generic TrainParity orchestration | production runner |
| 140-159, capture/serialization/comparison | 18 | generic TrainParity orchestration | production runner and snapshot worker |
| 160-170, four-state result core | 10 | generic TrainParity orchestration | production result model |
| 171-194, resource/timing evidence | 22 | Gate-4-only measurement except deterministic process evidence | audit derives measurements; production reports outcomes and PIDs only |
| 195-199, deterministic report write | 4 | generic orchestration/infrastructure | production atomic JSON report |
| 202-216, CLI and guard | 13 | generic resume-testing infrastructure | production API/CLI invocation; project caller becomes at most a thin file |

The whole 191-line file is generic or benchmark-only. It contains no ImageNet,
nanoGPT, or Ignite branch. Counting it as user glue was correct for Gate 4, but
leaving it there violates the low-friction product contract.

## pytorch/examples ImageNet

### Adapter

Source: `user_files/pytorch_examples_imagenet/trainparity_adapter.py`
(13 logical LOC).

| Physical lines / function | Classification | Disposition |
|---|---|---|
| 1-9, imports and `NAME` | project-specific training semantics | retain, with imports minimized |
| 11-12, `checkpoint_path` | project-specific checkpoint semantics | retain |
| 15-20, `normalize` | project-specific observed-state semantics | retain |

### Supporting glue

Source: `user_files/pytorch_examples_imagenet/trainparity_project_glue.py`
(62 logical LOC).

| Physical lines / function | Logical LOC | Classification | Disposition |
|---|---:|---|---|
| 1-12, imports/constants | 7 | mixed project semantics and generic wrapper imports | retain only split/total semantics; remove wrapper imports |
| 13-21, `prepare` | 6 | Gate-4-only tiny-data benchmark | audit fixture only |
| 22-36, `command` | 13 | generic resume-testing wrapper | replace with the adapter's explicit upstream command method |
| 37-65, `_worker` | 27 | project-specific upstream CLI plus generic mkdir/PID/`execv` | retain only explicit upstream argv; production owns directory, PID, launch and process boundary |
| 66-74, CLI guard | 9 | generic resume-testing infrastructure | production worker |

## nanoGPT

### Adapter

Source: `user_files/nanogpt/trainparity_adapter.py` (13 logical LOC).
Imports/name, `checkpoint_path` (lines 11-12), and `normalize` (15-20) are all
project-specific checkpoint/observed-state semantics and remain explicit.

### Supporting glue

Source: `user_files/nanogpt/trainparity_project_glue.py` (71 logical LOC).

| Physical lines / function | Logical LOC | Classification | Disposition |
|---|---:|---|---|
| 1-15, imports/constants | 9 | mixed project semantics, benchmark imports, generic wrapper imports | retain only split/total and command semantics |
| 16-25, `prepare` | 8 | Gate-4-only tiny memmap generation | audit fixture only |
| 26-39, `command` | 12 | generic resume-testing wrapper | production runner invokes explicit project command |
| 40-74, `_worker` | 33 | project-specific nanoGPT argv plus generic mkdir/PID/`execv` | retain only explicit argv semantics |
| 75-83, CLI guard | 9 | generic resume-testing infrastructure | production worker |

## Ignite MNIST engine example

### Adapter

Source: `user_files/ignite_mnist_engine/trainparity_adapter.py` (15 logical
LOC). Imports/name, `checkpoint_path`, and `normalize` are project-specific.
The sibling `rng_state.pt` observation is explicit because the upstream example
does not include RNG in its checkpoint; the benchmark driver records it without
replacing the upstream checkpoint implementation.

### Supporting glue

Source: `user_files/ignite_mnist_engine/trainparity_project_glue.py`
(65 logical LOC).

| Physical lines / function | Logical LOC | Classification | Disposition |
|---|---:|---|---|
| 1-18, imports/constants | 12 | mixed project semantics, benchmark loader imports, generic wrapper imports | retain only split/total and checkpoint semantics |
| 19-22, `prepare` | 2 | Gate-4-only no-op fixture hook | remove |
| 23-36, `command` | 12 | generic resume-testing wrapper | production runner |
| 37-46, `_load_example` | 8 | Gate-4-only import of the upstream example for a tiny fixture | audit driver only |
| 47-56, `_tiny_loaders` | 8 | Gate-4-only deterministic data | audit driver only |
| 57-72, `_worker` | 14 | benchmark invocation of original save/load plus generic PID/reporting | audit driver retains loader substitution; production owns launch and PID |
| 73-81, CLI guard | 9 | generic resume-testing infrastructure | production worker/audit driver entry point |

## Quantified duplication

| Duplicate family | Per deployment | Deployments | Counted user LOC | Avoidable duplicate LOC |
|---|---:|---:|---:|---:|
| Identical clean-resume orchestrator | 191 | 3 | 573 | 382 beyond one copy; all 573 leave user code |
| Repeated project wrapper `command` + CLI guard | 21-22 | 3 | 64 | 42 beyond one conceptual implementation; all 64 leave user code |
| Repeated run-directory/PID/child-boundary handling inside workers | at least 3 | 3 | at least 9 | at least 6; all leave user code |
| Project checkpoint adapters | 13, 13, 15 | 3 | 41 | 0; these are intentionally project-specific |

At least 646 of the 812 counted user lines are generic duplicated
orchestration/infrastructure before considering the repeated imports and mixed
measurement fields. The remaining project glue also contains Gate-4-only tiny
data and measurement work. Gate 4B therefore targets one small adapter plus an
optional thin invocation file per project, while keeping all original upstream
checkpoint save/load paths and zero upstream modifications.

## Production boundary for Gate 4B

The new abstraction may know only an importable process-case protocol,
execution plans, file paths, commands, raw observed state, environments, and
four-state results. It must not contain repository names or branches for
ImageNet, nanoGPT, Ignite, Lightning, Transformers, or DeepSpeed. Tiny datasets,
fault injection, RSS/wall-time measurement, external cloning, and license checks
remain under `experiments/gate4b`.
