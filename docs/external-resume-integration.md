# External resume integration

This guide shows how to retain a command-oriented resume equivalence check for
an external PyTorch project. TrainParity runs the project's commands; it does
not provide a framework adapter or rewrite the training loop.

## Complete copyable example

Assume this layout:

```text
project-root/
├── project/
│   └── train.py
├── trainparity_case.py
└── run_trainparity.py
```

The example training command owns its checkpoint implementation. It saves the
model, optimizer, scheduler, logical step, and CPU RNG needed to resume:

```python
# project/train.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    torch.manual_seed(17)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    step = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["rng"]["torch_cpu"])
        step = int(saved["step"])

    while step < args.end_step:
        inputs = torch.randn(4, 2)
        targets = torch.randn(4, 1)
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        optimizer.step()
        scheduler.step()
        step += 1

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": {"torch_cpu": torch.get_rng_state()},
            "step": step,
        },
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
```

The user-owned case maps TrainParity's execution plan to that command and
selects semantically named checkpoint observations:

```python
# trainparity_case.py
from __future__ import annotations

import sys
from pathlib import Path

import torch

from trainparity.api import ProcessExecutionPlan


class Case:
    name = "external_example"
    split_step = 2
    total_step = 4

    def command(self, plan: ProcessExecutionPlan) -> list[str]:
        command = [
            sys.executable,
            str(plan.cwd / "project" / "train.py"),
            "--checkpoint",
            str(self.checkpoint_path(plan.run_dir)),
            "--end-step",
            str(plan.end_step),
        ]
        if plan.resume_from is not None:
            command.extend(("--resume", str(plan.resume_from)))
        return command

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "checkpoint.pt"

    def observe_checkpoint(self, path: Path) -> dict[str, object]:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        return {
            "model": saved["model"],
            "optimizer": saved["optimizer"],
            "scheduler": saved["scheduler"],
            "rng": saved["rng"],
            "step": saved["step"],
        }
```

Run the check from a separate file so the case remains importable in every
fresh snapshot process:

```python
# run_trainparity.py
from __future__ import annotations

import json
from pathlib import Path

from trainparity import Outcome, check_resume

ROOT = Path(__file__).resolve().parent
result = check_resume(
    "trainparity_case:Case",
    cwd=ROOT,
    work_dir=ROOT / ".trainparity-runs",
    report_path=ROOT / "trainparity-report.json",
)
print(json.dumps(result.to_dict(), indent=2))
raise SystemExit(0 if result.outcome is Outcome.PASS else 1)
```

With the three files in place:

```bash
python run_trainparity.py
```

## Execution phases and checkpoint staging

`ProcessExecutionPlan.phase` is exactly one of:

- `baseline_a`: first uninterrupted run through `total_step`;
- `baseline_b`: independent uninterrupted repeat for self-consistency;
- `candidate_split`: run through `split_step`, save, and exit;
- `candidate_resume`: fresh process loading the staged split checkpoint and
  continuing through `total_step`.

`checkpoint_path(run_dir)` is a deterministic *location contract* for every
phase. TrainParity calls it after each child completes to locate the result.
It also calls it for `candidate_resume` before that child starts, copies the
`candidate_split` checkpoint there, and supplies that same path as
`plan.resume_from`:

```text
candidate_split/checkpoint.pt
        │ copy before resumed child launch
        ▼
candidate_resume/checkpoint.pt ── plan.resume_from
```

Consequently, `checkpoint_path()` must not assume that it is only searching
for an already-created output. Callback exceptions such as
`FileNotFoundError` become an `ERROR` result with the phase and boundary named.
The in-process `command()` and `checkpoint_path()` boundaries catch
`Exception`, not `BaseException`, so `KeyboardInterrupt` and `SystemExit`
continue to propagate. `observe_checkpoint()` executes in a snapshot child;
ordinary callback exceptions are `ERROR`, and abnormal child termination is
reported as a worker `ERROR`.

## Timestamped or implicit checkpoint paths

Some projects write paths such as
`saved/models/<timestamp>/checkpoint.pth`. Keep the upstream code unchanged
and put its nondeterministic layout behind a deterministic launcher. The
launcher receives the phase run directory, invokes the upstream command, and
then copies or symlinks the produced file to `run_dir/checkpoint.pth`:

```python
# checkpoint_launcher.py (project-specific glue)
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--end-step", required=True)
parser.add_argument("--resume", type=Path)
args = parser.parse_args()

command = [
    "python",
    "project/train.py",
    "--save-root",
    str(args.run_dir / "saved" / "models"),
    "--end-step",
    args.end_step,
]
if args.resume is not None:
    command.extend(("--resume", str(args.resume)))
subprocess.run(command, check=True)

created = list((args.run_dir / "saved" / "models").glob("*/checkpoint.pth"))
if not created:
    raise FileNotFoundError("upstream command produced no checkpoint")
newest = max(created, key=lambda path: path.stat().st_mtime_ns)
shutil.copy2(newest, args.run_dir / "checkpoint.pth")
```

The case's `command()` invokes this launcher, passing `plan.run_dir`,
`plan.end_step`, and optional `plan.resume_from`. Its `checkpoint_path()`
always returns `run_dir / "checkpoint.pth"`. The staged split checkpoint is
there before a `candidate_resume` launch; after training, the launcher replaces
it with the resumed result. A symlink may be used instead of a copy when it is
portable and the target remains valid for the full check.

## Child logs and error results

Pass an explicit `work_dir` when logs need to survive the check. TrainParity
writes:

```text
<work_dir>/baseline_a/stdout.log
<work_dir>/baseline_a/stderr.log
<work_dir>/baseline_b/stdout.log
<work_dir>/baseline_b/stderr.log
<work_dir>/candidate_split/stdout.log
<work_dir>/candidate_split/stderr.log
<work_dir>/candidate_resume/stdout.log
<work_dir>/candidate_resume/stderr.log
```

For example, a nonzero resumed child returns a message ending with
`see candidate_resume/stderr.log under the preserved work_dir`. Reports do not
embed arbitrary stderr or environment values. Without an explicit `work_dir`,
the managed temporary directory is cleaned, so its logs are not promised to
remain available.

## Recommended observations

Choose observations from the project's actual resume contract. Consider:

- model parameters and buffers;
- optimizer state;
- scheduler state;
- AMP / gradient-scaler state;
- CPU RNG state;
- CUDA RNG state for each applicable device;
- data order, sampler, or cursor state;
- epoch and global step;
- controls such as optimizer type, scheduler type, and important configuration
  values.

Richer, semantically named observations make the first observed difference
more useful. TrainParity still reports evidence, not an inferred root cause;
an unobserved state can make a result incomplete.

## Expected cost

A check runs `baseline_a`, `baseline_b`, `candidate_split`, and
`candidate_resume`, plus snapshot workers. The split and resumed portions
together approximate another normal trajectory, so roughly three normal-run
equivalents before snapshot overhead is expected. Do not remove
`baseline_b`, reuse a training process, or bypass fresh-process loading merely
to reduce wall time: those boundaries establish self-consistency and resume
evidence.

Integration effort depends heavily on the upstream checkpoint interface.
Projects with implicit or timestamped paths need more orchestration glue than
projects with an explicit output path. The resulting check is most useful when
retained as a regression or CI test rather than used as a one-off checkpoint
diff.
