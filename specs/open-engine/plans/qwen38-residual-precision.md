# Exact wide residual addition

2026-10-01. Follow-up to [segmented down precision](qwen38-down-segment-precision.md).
This stage isolates the first remaining residual-add error and adds an opt-in
vector implementation of IEEE binary32 round-to-nearest, ties-to-even.

## Failing evidence

The preceding full-model capture has identical layer1 inputs at channel3390:
res1=-0.03440730646252632 and fo=0.005703141447156668. CPU FP32 addition and
FP64 addition rounded to FP32 both produce-0.028704164549708366, while the
NPU LN path returns-0.028704166412353516. The error survives layer2 and changes
the next BF16 norm. The primitive fixture repeats this pair with swapped
operands and signs, so all5120 residual outputs fail before the change.

The original three random RMSNorm inputs have805,841 and860 incorrect FP32
sums. These pass the original1e-6 residual-relative tolerance. The new opt-in
strict gate rejects every differing residual bit, and retains the norm gate.
Across18 strict cases, the baseline has11311 differing residual values.

## Implementation and TDD

`LN_RESIDUAL_RNE=1` is restricted to streamed widths. The default stays off.
`fp32_add_rne.h` aligns unsigned mantissas using fixed SIMD shifts and sticky
bits, adds/subtracts integer lanes, normalizes, and explicitly rounds to nearest
with ties to even. A lane abstraction lets the same production algorithm run
in a host test against an independent NumPy FP32 reference. Production uses
32-lane AIE integer operations; no neural operator runs on the host.

The header handles zeros, subnormals, overflow, infinities and quiet NaN
propagation. Host tests check200000 random bit pairs and289 boundary pairs;
NaNs are checked by classification, other outputs bitwise. NPU LN acceptance
is for finite model-range operands, not nonfinite normalization.

Initial RED tests covered missing strict-fixture support, missing arithmetic
and stale recipe cache dependencies. The strict NPU fixture fails on the
unchanged baseline. The first new adder fixes every residual bit but exposes
640 signed-zero mismatches in normalized output. The final variant restores
the sign of exact residual zero after normalization, including the weight
sign; that same unchanged fixture then passes. No threshold is relaxed.

Recipe source hashes now include the integer header; standalone specialization
also includes the mode. Scratch, weight format, DMA and physical FP32/BF16
interfaces remain unchanged. The new routine is confined to the residual
addition; statistics and other operators keep their prior arithmetic.

## Primitive and compatibility results

| Check | Result |
|---|---|
|Strict RMSNorm cases|18/18 pass|
|Residual sums across those cases|92160/92160 bit-exact|
|Real channel3390 pair and sign/order variants|exact|
|Repeated input after unrelated calls|bit-exact|
|Original norm thresholds/canaries|pass|
|Rebuilt old mode|12/12 pass, all y/xn captures bit-exact to prior build|
|CPU suite|824 passed,47 skipped|
|Text/core|6736 B|
|Data plus reserved stack/core|57600 B|

The guarded18-case diagnostic measured median dispatch0.216 ms for the new
kernel versus0.138 ms for the baseline. These short samples establish a cost
in this run, not a stable application benchmark. Keep the accuracy mode opt-in.

| Artifact | SHA256 |
|---|---|
|LN xclbin|`e4f9498e12b8f389f89abfb4ad9fd1c247b8a18c881f83565ec190db8f21be28`|
|LN instructions|`29fe75a6fbfc46032ec6adfef5cbbdd97d4cdd7ae42a3c0c16de019211eef05e`|

## Reproduction

Use [open-residual-rne](../../../.opencode/skill/open-residual-rne/SKILL.md).
Artifacts live in `open_kernels/designs/wide_deltanet/build_residual_precision`:
`baseline` retains the old failing strict fixture; `ln` the first adder with
the signed-zero norm failure; `ln_rne` the passing primitive; `regression` the
rebuilt old mode; `full` the unchanged-reference model replay.

## Full-model replay

The replay changes only LN relative to `build_down_precision/carry/full`.
It reuses the corrected output projection and segmented FFN, with all other
kernels, packed fixtures, reference tensors, tokens and tolerances unchanged.
All3564 calls complete. Token feedback remains248045 ->8678 ->198 ->2;
reset, state/cache copies and canaries pass. Minimum logits correlation is
0.9999872928, above0.9999.

All384 captured residual additions now match FP32 addition of their actual
device operands bitwise:0 differing values out of1966080, versus206956 in the
preceding replay. The extended `diagnose-wide-model-rounding.py` checks both
input-x+projout=res1 and res1+fo=y, without changing the independent trajectory.
Layer1 y3390 is now-0.028704164549708366 and layer3 xn3390 is
0.00011587142944335938, exactly their references.

Full numerical acceptance is nevertheless **FAIL**:18886/18896 slice and
4121/4122 decode checks pass. Failures increase from4 to11, even though residual
addition is now exact. Layer63 y and final norm maxrel improve, while other
bounds fail. This kernel is experimental and must not be promoted as the
accepted full-model default. PR4 still requires numerical closure, runtime
integration and benchmark work.

| Position | Field | maxrel | Required bound |
|---|---|---:|---:|
|cold-1-layer55|y|0.00739337685625|<0.005|
|cold-1-layer56|y|0.00678194532414|<0.005|
|cold-1-layer57|y|0.00706967730464|<0.005|
|cold-1-layer58|y|0.00560146855468|<0.005|
|cold-1-layer59|res1|0.0134321092544|<0.01|
|cold-1-layer59|y|0.00924080375773|<0.005|
|cold-1-layer60|res1|0.0243927566862|<0.02|
|cold-1-layer60|xm|0.0245398773006|<0.02|
|cold-1-layer61|y|0.00695396837391|<0.005|
|cold-1-layer63|y|0.0161538378897|<0.005|
|cold-1|final_norm|0.00854700854701|<0.008|

The first changed BF16 boundary is now cold0/layer3 post norm, with72 propagated
differences and no local norm errors. The first seven norm boundaries (through
layer3 entry) are exact. Across384 boundaries,831413 values differ, with26
local norm errors; these diagnostics overlap with propagated differences and
must not be added.

## Next isolated target

Layer3 is the first full-attention layer. Its BF16 xn is exact, while qg and kvn
have maximum FP32 errors9.5367431641e-7 and1.0132789616e-6. Its og has27 differing
BF16 values and maximum absolute error0.0001220703125; the output projection
then differs by up to1.7642974854e-5 and res1 by1.5258789063e-5.

At this first token, attention has one row, so softmax is exactly1. An offline
conditional check using the captured V (rounded to BF16) and sigmoid of the
captured gate has21 differences from device og and6 from the independent og.
Thus both projected inputs and attention arithmetic need isolation; correcting
the residual sum alone cannot remove this first attention discrepancy. Use
`wide_attention_reference.decode` and a guarded real-frame replay to distinguish
gate/V rounding from device attention math before replacing another kernel.
Conditional diagnostics must not replace the original model oracle.
