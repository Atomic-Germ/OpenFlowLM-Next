# PR4B bounded host-fold experiment protocol

This protocol is fixed before hardware correctness or performance collection.
Control base: `ad867144` (committed PR4A profiling). Branch:
`perf/npu-host-fold-dedup`. Historical evidence remains separate.

## Only candidate

The high-weight half validates the input and computes the BF16 residual once.
The low-weight half borrows that same host-owned residual until its two existing
dispatches complete. Remove only its duplicate split/validation loop; the initial
validation remains unchanged. Retain four products, all three output copies and
FP32 `(a+b)+(c+d)` grouping. No ownership redesign, output-copy removal, new
dispatch, kernel, allocation refactor, DeltaNet or cache arithmetic change.
The opt-in is OFF unless explicitly set to exactly 1. Defaults stay unchanged.

## Required correctness gates before timing

Production control/candidate complete-buffer CPU equality on adversarial FP32,
NaN/infinity/BF16-conversion overflow rejection, overwritten mapped outputs,
full/partial shapes, mixed-size workspace reuse, input identities and sum order.
Then full 64-layer hardware comparisons for 255/256/257 prompt tokens with exact
logits, greedy continuation and sampled linear/full-attention state. Include two
resident requests and reset/isolation. No relaxed tolerance: required compared
buffers and output IDs must be byte-exact. Stop timing on any correctness failure.
Shared-engine build/startup and existing integration gates must be reported;
CLI-only acceptance is not a full application acceptance claim.

## Measurement protocol

- Same model, kernel files, XRT core/coreutil/xdna plugin hashes and power policy.
  Use the same freshly attested candidate-capable executable OFF/ON for paired
  comparisons; separately verify OFF against the unchanged control executable.
- Primary scenarios: 256, 512, 1024 real prompt tokens (one generated output),
  and short decode: 32 prompt tokens plus 32 generated steps. All use 64 layers.
- One excluded warm-up pair then five matched pairs per scenario, alternating
  OFF/ON order. Collect direct fold profile in both prefill arms; identical
  optional layer profiling is allowed in both arms. State clearly whether primary
  wall samples include profiling. No synthetic-to-production extrapolation.
- Capture per-process sampled RSS/high-water RSS, individual timings, full launch
  environment/argv, input/artifact hashes and observed loaded runtime mappings.
- Around-4K real-context decode is optional; do not emulate populated context
  with seek or claim it measured if excluded for campaign cost/capacity.
- Predeclare power before the first timed run and keep it fixed. An unavailable
  control must be reported, not silently substituted. Retain thermal/load data.

## Fixed decision rule

Primary gain is the median of paired percentage reductions in 256-token prefill
wall, not a difference of medians. Target is at least 1.00%.

MERGE CANDIDATE requires all numerical/integration gates, at least 1% primary
paired-median gain, at least four of five initial primary pairs improving,
no greater than 1% paired-median regression at 512/1024 or short decode, and
no greater than 1% (or 32 MiB, whichever is larger) RSS increase. A result whose
paired difference spread overwhelms its median is not established reproducibility.

Only one bounded confirmation is allowed: if the initial primary gain is at least
1% but variability or only three wins leaves it ambiguous, collect exactly ten
additional 256-token pairs after one excluded warm-up pair. Declare KEEP DRAFT
until that confirmation is reviewed. If run within this experiment, confirmation
must itself have at least 1% paired-median gain and at least eight of ten wins;
both initial and confirmation sets remain separate. Do not pool or extend until
the target is reached. Initial gain below 1%, correctness failure, or meaningful
regression is PARK / NO-GO without further candidate tuning.

No production-default promotion, push, merge, rebase, or additional optimization
is authorized by this protocol.
