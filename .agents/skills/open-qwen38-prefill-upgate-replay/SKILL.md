---
name: open-qwen38-prefill-upgate-replay
description: Replay captured Qwen3.8 hybrid FFN up/gate with production GEMM and paired BF16 input, require exact native identity, and isolate weight-rounding error without model residency.
---

# Isolated hybrid up/gate replay

Use for the L63/P7 projection identified by causal sublayer/vector attribution.
Follow the [runbook](../../../utilities/prefill-upgate-replay/README.md) for the
standard harness build, preparation and serial NPU dispatches. No kernel build
or full resident model is needed. Keep the production export and retained
captures intact; create fresh preparation/build directories.

The preparation revalidates retained capture/build/reference hashes, effective
source weights and the previous production pool. The up/gate matrix is
N34816 x K5120, concatenated up then gate. Preserve the manifest's RS2 packing.
T is 256; the actual target uses slot 7. Slot 135 is a second placement control.
All other rows are zero, and this substitution is only accepted after the
selected result reproduces the retained native capture bytewise.

Use the same paired-input arithmetic as the private native CLI: high is
BF16_RNE(x), low is BF16_RNE(FP32(x-BF16_RNE(x))). Add the two GEMM outputs in
FP32, never FP64. Both individual outputs must have zero inactive columns.
Reject partial/nonfinite arrays, stale source/build hashes and any bitwise
control mismatch before interpreting numerical diagnostics.

Independent references use effective Q4 weights with original FP32 input,
then paired input. A separate BF16-rounded-weight reference diagnoses the
kernel's dequantization boundary; it must not replace the exact-weight primary
reference. Keep separate up/gate maxrel <0.005 and rel_fro <=0.005 gates.
Control validity is not full-model acceptance.

## Validated evidence

2026-10-20: four dispatches complete; both layouts reproduce all 34816 captured
values exactly. Packed/source effective weight identity passes. Primary up/gate
maxrel is 0.0019482680 / 0.0002586737. Against BF16-rounded weights and paired
input, it is 0.0000011273 / 0.0000015830. The CPU rounding effect is
0.0019497818 / 0.0002588257, accounting for most of the selected projection error.

GCC 15.2.0, CMake Release, local XRT SDK; 326 compiler dependencies pinned.
Harness SHA256: acd59b215f0276e80c667e8e3cc9c67bd24cabdbde38342073ee82c7887ca74d.
Reuse production-consistent build key
ac29fbcd21859648ff9d59f79e77f018b99daec0888f77a693e7ff18b2237032.
Logical per-process NPU buffers are about 0.1395 GiB excluding XRT overhead.
Evidence: Models/qwen38-27b/prefill-upgate-replay/ (run, harness and logs).
Tests: 1513 passed / 47 skipped, 13 new RED-to-GREEN cases.

Next test an isolated higher-precision weight-dequantization/GEMM candidate
against this byte-exact control. Up and down errors partly cancel in the
propagated FFN, so improved projection accuracy alone is not model acceptance.
Repeat independent causal gates before promotion. Sequential upstream-history
failures remain separate. Full-model long-prefix accuracy still fails.
