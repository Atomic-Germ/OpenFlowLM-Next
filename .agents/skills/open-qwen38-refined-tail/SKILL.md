---
name: open-qwen38-refined-tail
description: Build a private full-ax forced-OG replay and validate refined L3 attention through production output projection and FFN at P125/P205/P239 with exact native controls.
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
  and scratch and the original current layer input. Positions are fixed at
  125, 205 and 239; no previous candidate-layer output is fed into another case.
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
Next extend the forced-tail validation to all 240 causal positions before
private full-ax projection-refinement integration.
