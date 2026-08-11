# Gate 5 report

## Outcome

**PASS — recommendation: GO**

Gate 5 validates only user-declared full-batch/microbatch equivalence over one
optimizer-update boundary. Human review is required before Gate 6.

## Results

- Clean false positives: 0
- Stable CPU faults: 7/7 over 3 repeats
- Same-device GPU faults: 8/8 including AMP timing, NVIDIA L40S, job `58980407`
- Verified-equal initial state and three distinct fresh PIDs: yes
- Peak temporary-directory disk: 23237352 bytes
- Persisted recorded evidence: 69620 bytes
- Hosted CI: run `31453089659`, conclusion `success`

## First observed divergences

- `missing_accumulation_scaling`: `loss_accounting` at `loss_accounting.denominator`
- `variable_length_mean_of_means`: `loss_accounting` at `loss_accounting.denominator`
- `optimizer_step_per_microbatch`: `gradient` at `gradient.weight`
- `scheduler_step_per_microbatch`: `optimizer_state` at `optimizer.param_groups[0].lr`
- `zero_grad_wrong_time`: `gradient` at `gradient.weight`
- `gradient_clip_wrong_time`: `gradient` at `gradient.weight`
- `amp_unscale_scaler_timing`: `gradient` at `gradient.weight`
- `incomplete_final_window`: `loss_accounting` at `loss_accounting.denominator`

These are first observed divergences, not root-cause claims.

## Product surface

- `pytorch_examples_imagenet`: 38 user logical LOC, upstream modified LOC 0, PASS
- `nanogpt`: 37 user logical LOC, upstream modified LOC 0, PASS

Both checks use fresh clones pinned to the Gate 4 commits, add two user files,
modify zero upstream lines, and remain below 50 logical LOC. The ImageNet clean
relation explicitly fixes BatchNorm in eval mode. Its retained training-mode
control fails first at `loss_accounting.effective_loss`, demonstrating that
full-batch/microbatch equivalence is not universal. Dropout is likewise not
claimed equivalent without explicit user semantics. nanoGPT's tied parameter
mapping first returns `ABSTAIN`; the product fixture explicitly chooses an
unambiguous optimizer subset without weakening production mapping.

## Policy and scope

Loss numerator/denominator accounting is optional and unavailable accounting is
reported, never inferred. ExactComparison and the fixed explicit tolerance
policy remain separate; tolerance is never inferred or tuned from differences.
Complex batches require an explicit splitter unless the safe ordered tensor-tree
split applies. FullValueBackend remains the correctness reference. Gate 6,
sample coverage, distributed support, framework adapters, and services were not
started.
