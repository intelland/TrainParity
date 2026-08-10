# Gate 4 external integrations

Gate 4 evaluates experiment-only adapters against three real, external PyTorch
repositories. No upstream source is copied into TrainParity and no tracked line
in any upstream checkout is modified. The generated image and token data are
untracked files inside the isolated M3 Gate 4 area.

## Pinned repositories and licenses

| Integration | Repository | Exact commit | License | License file SHA-256 |
|---|---|---|---|---|
| `pytorch_examples_imagenet` | `https://github.com/pytorch/examples.git` | `acc295dc7b90714f1bf47f06004fc19a7fe235c4` | BSD-3-Clause | `7e5dc9b5cf276166c4a0678d33e53b14e3d7ef47eee9356ab786c4ba45414efd` |
| `nanogpt` | `https://github.com/karpathy/nanoGPT.git` | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | MIT | `a59ec5cdb1c1e447e5266a014b3e7ded511f8d6e5a08931e931c87490ff821fe` |
| `ignite_mnist_engine` | `https://github.com/pytorch/ignite.git` | `e08ff9257ed18d8d805304e32ba85a44553195fc` | BSD-3-Clause | `4b026f6919b2d6a97e93dfe72f85a7111e3f3834cc17d7d6a0ff475e4a22720c` |

The license identifiers are based on each pinned repository's root `LICENSE`
file. The matrix verifies both the commit and license hash before accepting a
project result.

## Original checkpoint implementations

- `pytorch_examples_imagenet` executes the pinned
  `imagenet/main.py`. Checkpoints are created by its `save_checkpoint` call and
  loaded through its original `--resume` `torch.load` and `load_state_dict`
  path.
- `nanogpt` executes the pinned `train.py`. It creates `ckpt.pt` with its
  original `torch.save(checkpoint, ...)` call and resumes through
  `--init_from=resume` and the original `torch.load` path.
- `ignite_mnist_engine` imports and invokes the pinned
  `examples/mnist/mnist_save_resume_engine.py`. It saves through Ignite
  `Checkpoint` and `DiskSaver`, then loads through the example's original
  `torch.load` and `Checkpoint.load_objects` calls.

PyTorch 2.6 changed the default `torch.load` policy. The pinned Ignite example
predates that change and stores NumPy RNG state, so Gate 4 sets
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for this trusted local checkpoint. The
upstream load call itself remains unchanged.

## State selection and faults

Each normalizer selects ordinary continuation state from the complete upstream
checkpoint; none branches on the injected fault.

- ImageNet compares epoch, model, optimizer, and scheduler state. The injected
  fault decrements the saved scheduler epoch.
- nanoGPT compares model, optimizer, model arguments, iteration, and best
  validation loss. The injected fault decrements the saved iteration.
- Ignite compares trainer Engine, model, optimizer, and scheduler state. The
  injected fault decrements the scheduler epoch. The upstream checkpoint's
  `RunningAverage` is excluded: the Engine resets this reporting-only derived
  metric after load, and it does not determine subsequent updates. The clean
  prototype exposed this behavior before the state boundary was documented.

Integer optimizer checkpoint keys and NumPy arrays are converted by shared
experiment glue to deterministic string keys and immutable
`{dtype, shape, values}` records before the Gate 2 full-value backend freezes
them. This conversion does not change the production optimizer name-mapping
contract and is not promoted as a framework adapter.

## Integration effort

Logical LOC means nonblank, noncomment lines and is counted automatically.

| Integration | Adapter LOC | Supporting glue LOC | Upstream modified LOC | Total newly written project LOC |
|---|---:|---:|---:|---:|
| `pytorch_examples_imagenet` | 24 | 53 | 0 | 77 |
| `nanogpt` | 24 | 57 | 0 | 81 |
| `ignite_mnist_engine` | 24 | 44 | 0 | 68 |

All three totals exceed 50 lines, although the median adapter is 24 lines:

- `pytorch_examples_imagenet` needs deterministic tiny ImageFolder generation,
  command-line construction, process replacement, and resume-path routing. The
  training and checkpoint algorithms remain upstream.
- `nanogpt` needs a tiny binary token dataset and metadata plus explicit
  upstream configuration arguments that shrink the real model and disable
  compilation and logging. No nanoGPT training logic is reproduced.
- `ignite_mnist_engine` needs dynamic loading of the exact external example, a
  four-sample loader replacement, and deterministic selection of the
  epoch-numbered checkpoint emitted by the upstream saver.

Shared matrix orchestration is reported separately and is not concealed in the
24-line adapters. It creates continuous references, copies the original split
checkpoint, invokes clean and faulty resumes, snapshots results, records every
difference at the first divergent step, verifies repository state, and measures
resources.

## Hand-written control

The minimal hand-written comparator only recursively tests final model equality.
It is shorter, but its only diagnostics are `final model states are equal` and
`final model states differ`. It can miss faults that alter non-model resumable
state and cannot identify a first step or state path. TrainParity records the
first observed divergence and all differences at that step; it does not claim
that the first observation is the root cause.

## Scope

The runs use generated tiny data and one A100. Gate 4 measures integration
friction and diagnostic value, not model quality, scale, distributed execution,
or gradient accumulation. The full-value reference backend is deliberately not
optimized in this gate.
