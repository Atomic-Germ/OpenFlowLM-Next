---
name: open-qwen38-refined-tail
description: Build a private full-ax forced-OG replay and validate refined L3 attention through production output projection and FFN at selected positions and across all 240 positions with exact native controls.
---

# Refined native attention through the complete L3 tail

Read `utilities/l3-tail-replay/README.md` for commands and the measured result.
Use this after the independently audited `l3-attention-replay` experiment.
It is a forced-boundary diagnostic, not an integrated correction or a full-model
run. The refined inputs include two NPU projection passes plus a host FP32 sum.

## Build and run

1. Source `utilities/env.sh`; set `OPENBLAS_NUM_THREADS=1`.
2. Build `utilities/l3-tail-replay/build.py --spec <export>/spec.json --out <fresh Models directory>`.
   The builder generates current recipe-specific translation units privately,
   preserving all production arithmetic. Only the output projection's input
   DMA changes, to a new seventh data-buffer argument for 6144 FP32 OG values.
3. Prepare with `replay.py prepare --source <audited native attention run>
   --build <private build> --out <fresh run>`.
4. Run with `replay.py run --out <run> --harness <audited run_kernel>`.
   The tool uses the existing exclusive device lock. One L3 weight pool is
   resident (232,783,872 bytes); no full-model residency is needed.
5. Independently run `replay.py audit --out <run>`. Failed numerical gates must
   remain failures even if execution and audit are complete.

## Critical contracts

- Never append OG to the position-table argument. `attnpos` classifies every
  transfer from argument 5 as a position-record fill and correctly rejects
  duplicates. Preserve that guard; use argument 6 for external OG.
- Preserve the original six argument positions, runtime `attngeom`, original
  ptab and KV dimensions, and production position patching.
- Private generated units need the production layer_x helper include path
  (for `dnx_factored.h` and related shared helpers). Check actual compiler
  dependencies, generated/copy bytes, linked ELFs and allocated MLIR.
- Each control/candidate dispatch restores the original preceding KV prefix
  and scratch and the original current layer input. The selected replay fixes
  positions at 125, 205 and 239; the sweep uses 0–239. No previous
  candidate-layer output is fed into another case.
- Require all nine full native controls bytewise (act, state, xres at all three
  positions) before interpretation. Candidate upstream Q/gate/K/V/OG and KV
  must also match the original captures exactly.
- Scratch AA_OG is the recomputed ORIGINAL attention output. The actual refined
  input is `candidate-og_<position>.bin`. Do not mislabel the scratch capture.
- Primary full-layer references use independent unrounded-norm causal attention
  followed by FP64 output projection, residual, post-norm and FFN. Separately
  retain original full-model residual comparisons and conditioned output/FFN
  gates, including each split-down partial. Existing 0.005 bounds stay fixed.
- Save numerical arrays and replay their dtype, shape and bytes during audit.
  Report per-position regression independently of threshold PASS; keep P239.

## Build evidence

`Models/qwen38-27b/post-prefill-drift/l3-tail-build-v3` has 15 cores,
maximum .text 13,760/16,384 B, L1 end 59,920/65,536 B and unchanged 6144 B
stacks. 858 compiler inputs are pinned. The missing-helper-include build and
the duplicate-position-table replay were retained as unsuccessful attempts;
never relabel their evidence as successful or weaken checks to accept them.

The native run is `Models/qwen38-27b/post-prefill-drift/l3-tail-v2`.
Full-prefix propagation, integrated projection refinement, model numerical
acceptance, request isolation and production performance require separate
validation. No model registry, installed service or production ABI is changed.

Verified selected-position result: all nine native controls match; primary and
conditioned output/FFN gates pass at all three positions. Independent primary
full-layer maxabs decreases 21.75× / 8.08× / 1.23× at P125/P205/P239. Retained
full-model reference errors also improve at all three. The P239 attention-only
regression must still remain documented; downstream residual improves here.
Separate audit binds 18 captures and three NPZ files, reproducing 39 arrays
bytewise. Ten new tests; full host suite 2036 passed, 47 skipped.

## Full-prefix sweep

Reuse the audited binary and selected-position run; do not edit their pinned
sources or overwrite captures. `utilities/l3-tail-replay/sweep.py` provides:

```bash
python utilities/l3-tail-replay/sweep.py prepare \
  --source <audited attention run> --baseline <audited selected tail run> \
  --build <same private build> --out <fresh sweep>
python utilities/l3-tail-replay/sweep.py run --out <sweep> --harness <audited run_kernel>
python utilities/l3-tail-replay/sweep.py audit --out <sweep>
```

Keep `OPENBLAS_NUM_THREADS=1` for exact numerical overlap. This runs 480
single-layer dispatches. At position zero both trajectories load full zeroed
state and scratch; all other positions restore the original preceding state.
Require all 720 native controls, 18 raw selected-position overlaps and 39
selected numerical-array overlaps. Audit all 1440 raw captures and 240 NPZ
files, recomputing all 3120 arrays. Report improved, regressed and tied positions
separately for the primary and full-model references. A tolerance PASS is not
a claim of universal improvement. No candidate residual propagates to another
case; integrated projection refinement needs its own validation.


Verified sweep: `Models/qwen38-27b/post-prefill-drift/l3-tail-sweep-v1`.
All 480 dispatches, 720 native controls and 240 local gate cases pass. Audit
reproduces 3120 arrays across 1680 artifacts. Primary maximum maxabs falls
2.08×, but 178 positions improve, 61 regress and one ties. Preserve P46
(largest absolute increase), P148 (largest ratio) and P239 (earlier attention
regression). Full-model-reference comparisons have 155 improvements, 83
regressions and two ties. No tolerance was relaxed. Seven new TDD tests;
full host suite 2043 passed, 47 skipped. Next validate private integrated
refinement against all retained positions; model acceptance remains FAIL.
