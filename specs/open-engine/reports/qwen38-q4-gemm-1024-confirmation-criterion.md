# Predeclared acceptance criterion — 1024-token prefill confirmation

Written **before** collecting any data for this experiment, on 2026-11-15, after
the acceptance matrix in `qwen38-q4-gemm-acceptance.md` had already failed its
1024-token gate at −0.60% with 4/5 pairs.

This file fixes the decision rule so the result cannot be reinterpreted after
the fact, and so repetitions cannot be added until a favourable answer appears.

## Question

Is the 1024-token prefill improvement of the opt-in DMA-batching candidate
reproducible, or was the earlier −0.60% a noise-dominated measurement?

## Fixed design

- Arms and artifacts unchanged: `model-batch1` (control) vs `model-batch4`
  (candidate). Only the 20 GEMM kernels differ; verified again before the run.
- One shared warm-up pair (`rep = -1`), excluded from all statistics.
- **14 measured pairs, fixed in advance.** No additional pairs will be added,
  whatever the outcome.
- Schedule: the harness alternates order by repetition parity, giving
  A/B, B/A, A/B, B/A … — balanced, with each arm leading seven times.
- Prompt: 1024 tokens, 16 generated tokens, same binary, same method as the
  acceptance matrix (`omp_policy.py --tokens 1024 --reps 14`).
- Host load, thermal zones, governor, runtime environment and context capacity
  recorded around the run by the existing harness and a passive capacity probe.

## Statistics

For each pair `i`, `d_i = after_prefill_i − before_prefill_i` (negative = faster).

Reported: per-pair differences, median relative improvement, the median of
`d_i`, its interquartile range, the 14-pair sign count, a one-sided exact sign
test, and a 95% bootstrap confidence interval (10 000 resamples, fixed seed) for
the median of `d_i`.

## Decision rule (all must hold for PASS)

1. **Median relative improvement ≥ 1.00%.**
2. **At least 12 of 14 pairs faster.** One-sided sign test: p = 0.0065, so this
   also excludes "no improvement" at α = 0.01.
3. **Bootstrap 95% CI for the median of `d_i` entirely below 0** (the whole
   interval indicates a real saving, not just a favourable point estimate).
4. **Decode median change ≤ +3%** and **RSS median change ≤ +3%** (existing
   gates, unchanged).

If any condition fails → **improvement not established** → PARK the candidate:
stop optimizing this DMA approach, keep it opt-in and unpromoted, and move to
the next bottleneck from the existing profiling report.

If all hold → the improvement is established at this length; finalize the
performance report and recommend merge **as opt-in only**, with the production
default unchanged unless separately approved.

## Explicitly out of scope

No change to DMA batching, XRT, firmware, driver parameters or model precision.
No re-build of either arm. No additional lengths, no additional repetitions.
