# Gate 4 Friction Audit

**Audit outcome:** PASS

This is a Gate 4 rework audit only. Gate 5 was not started.

## Fresh-clone user cost

| Project | Adapter LOC | Supporting glue LOC | Total user LOC | Upstream LOC | Clean | E2E multiplier |
|---|---:|---:|---:|---:|---|---:|
| pytorch_examples_imagenet | 13 | 253 | 266 | 0 | PASS | 15.91x |
| nanogpt | 13 | 262 | 275 | 0 | PASS | 4.03x |
| ignite_mnist_engine | 15 | 256 | 271 | 0 | PASS | 3.85x |

Every row was reproduced from a new exact-commit clone, a no-dependencies wheel install, and only the three listed user source files. Generated data, checkpoints, logs, and the isolated wheel target are runtime artifacts, not hidden integration source.

The supporting glue exceeds 50 LOC in every case because the current production API does not orchestrate command-oriented external repositories. The audit counts process launch, baseline repetition, checkpoint staging, snapshot serialization, and reporting rather than hiding them in experiment helpers.

## End-to-end measurements

| Project | Normal | Baseline self-check | Save/exit + new-process/load/resume | Snapshot | Serialize | Compare | Total | Peak RSS KiB | Artifacts bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch_examples_imagenet | 4.7254s | 9.5003s | 9.0983s | 56.3392s | 0.1616s | 0.004680s | 75.1699s | 1153264 | 122107391 |
| nanogpt | 3.9731s | 7.9961s | 7.8956s | 0.0779s | 0.0066s | 0.000232s | 15.9922s | 1032112 | 183627 |
| ignite_mnist_engine | 5.2754s | 9.9995s | 9.2986s | 0.9653s | 0.0134s | 0.001254s | 20.2908s | 1110560 | 4064688 |

The earlier comparator-only timing is preserved in the JSON report and is explicitly not the total TrainParity overhead. Phase aggregates overlap where labeled: baseline self-consistency includes both normal runs, while snapshot, serialization, and comparison are also shown separately.

## Hand-written controls

The existing 12-line final-model-only comparator is retained as a **weak baseline**. It omits optimizer, scheduler, RNG, process orchestration, and path-level diagnostics.

The closer hand-written Ignite test is project-specific and has 116 logical lines (plus the explicitly reported user glue dependency). It checks model, optimizer, scheduler, and torch CPU RNG across a fresh-process resume. Its fault diagnostic is: `first observed divergence at $.scheduler.last_epoch`.

## Fault classification

| Project | Classification | First observed divergence | Downstream parameters diverged |
|---|---|---|---|
| pytorch_examples_imagenet | control-state | `scheduler.last_epoch` | false |
| nanogpt | trajectory-affecting | `best_val_loss` | true |
| ignite_mnist_engine | control-state | `lr_scheduler.last_epoch` | false |

These are classifications and first observed divergences, not root-cause claims. No reporting-only fault was injected in the accepted three-project suite.

## Preservation and scope

- Accepted Gate 0-4 evidence files checked: 22
- User document SHA-256 (local pre-run observation): `6b532b94660949abec3c50bbe826d2154a797eeec744d3e5c91058dec9a96300`
- Production API changes: none
- New framework adapters: none
- Snapshot backend optimization: none
- New projects: none
- Gate 5 work: none

## Exact commands

See `projects[].exact_commands` in the JSON report for every clone, checkout, install, import, and clean-resume command. Verification commands are also recorded there.

## Recommendation

REWORK: the technical GO evidence remains positive, but the honest user-required glue is well above 50 LOC because command-oriented orchestration is not in the production API. Do not begin Gate 5; human review should decide whether this friction is acceptable.
