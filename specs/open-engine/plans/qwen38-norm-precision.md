# Real RMSNorm statistics and product rounding

2026-10-02. Follow-up to [attention rounding](qwen38-attention-boundary.md).
The isolated cold0/layer4 input norm now matches the independent model in
all5120 BF16 values. The new arithmetic remains an opt-in experiment;
primitive acceptance does not establish full-model support.

## Failing evidence

The guarded baseline replay reproduces one local mismatch at channel786,
with no propagated norm differences. The real input is0.06631811708211899
and weight1.0703125. Device output is0.059814453125, while both conditional
and model references are0.0595703125. The unrounded independent value is
0.059692375144818, only7.667682e-9 below the BF16 midpoint0.0596923828125.

An opt-in trace kernel exposes the32 statistics lanes, their compensation,
sum, mean, inverse root and selected pre-BF16 products. It is diagnostic only:
it replaces xn with FP32 trace values and cannot satisfy norm acceptance.

| Quantity | Baseline device | Independent FP64 |
|---|---:|---:|
|Sum of squares|7239.64794921875|7239.648881883223|
|Mean plus epsilon|1.4139946699142456|1.413994922242817|
|Inverse root|0.8409615159034729|0.8409614248914107|
|Input times weight|0.07098111510276794|0.07098110968945548|
|Pre-BF16 output|0.0596923865377903|0.059692375144818|

The corrected trace has sum7239.64892578125, mean1.4139949083328247,
inverse root0.8409613966941833, weighted input0.07098110765218735 and
pre-BF16 output0.059692371636629105. These are on the correct side of the
BF16 boundary; the final FP32 value still differs from the FP64 oracle rounded
once to FP32. Trace repeats and residual outputs are exact in both variants.

The sum of captured lanes in FP64 is7239.648299694061: both lane accumulation
and the final reduction contribute. The weighted product is also above its
correctly rounded FP32 value. Fixing only residual addition cannot remove
these errors.

## Implementation and TDD

`LN_NORM_RNE=1` requires compensated wide LN and exact residual addition.
It uses a generic integer-lane binary32 multiplier, splitting each24-bit
significand into12-bit halves to retain the full48-bit product, then applying
normalization, sticky shifts and ties-to-even rounding. The same production
algorithm runs with scalar integer lanes in host tests and vector lanes on NPU.
It handles subnormals, signed zeros, overflow, infinities and NaN classification.

The new mode uses exact FP32 products for squares and normalization scaling,
exact add/sub for the existing Kahan lane sums, and a compensated32-lane
reduction that includes the saved corrections. The scalar mean and reciprocal
root remain unchanged. Intermediate FP32 rounding remains; this is not an
implementation of the FP64 oracle or a promise of globally exact BF16 norms.

Tests first failed for the absent real-frame diagnostic utility, multiplier
header and recipe cache dependencies. The multiplier then passed300000 random
bit pairs and289 boundary pairs against NumPy FP32, checking NaNs by class and
all other results bitwise. A strict real-frame test rejects even one differing
BF16 value, validates fixture hashes, and distinguishes conditional versus model
error. No independent full-model reference or threshold was changed.

The first build duplicated static arithmetic routines in the two workers and
overflowed program memory. Shared noinline COMDAT definitions remove that
duplication. This is confined to opt-in streamed LN; recipe dependencies and
standalone specialization hashes include the new headers and modes.

## Hardware and compatibility

| Check | Result |
|---|---|
|Existing strict residual RMSNorm cases|18/18 pass|
|Real layer4 norm|5120/5120 BF16 exact, repeated run exact|
|Rebuilt previous mode|18/18 pass, all36 y/xn captures bit-exact|
|CPU regression|831 passed,47 skipped|
|Text/core|11536 B|
|Data plus reserved stack/core|57600 B|
|DMA|Unchanged|

The18-case primitive fixture has one BF16 norm difference under both modes;
its inherited acceptance still passes. The real-frame strict gate is separate.
Median dispatch in this short run is0.8625 ms versus0.2125 ms for the prior
mode. This demonstrates additional cost, not a stable application benchmark.

| Artifact | SHA256 |
|---|---|
|LN xclbin|`cb734096b65366908636f974af4c89dddd5b9cc8a181705e19a2223683bb7160`|
|LN instructions|`29fe75a6fbfc46032ec6adfef5cbbdd97d4cdd7ae42a3c0c16de019211eef05e`|

## Reproduction

Use [open-norm-rne](../../../.opencode/skill/open-norm-rne/SKILL.md).
Artifacts are under `open_kernels/designs/wide_deltanet/build_norm_boundary`:
`baseline`, `trace`, `trace_frame`, `rne`, `rne_frame`, `regression`, `rne_trace`,
`rne_trace_frame` and `full`. Both traces deliberately retain `passed: false`;
their compare exits1 because they are not numerical acceptance artifacts.
Logs are `/tmp/norm-{rne,regression}-{build,hardware,compare}.log`,
`/tmp/norm-{trace,rne-frame}-hardware.log`, `/tmp/norm-unit.log` and
`/tmp/norm-full-prepare.log`; full hardware output lives in `full/hardware.log`.

## Full-model replay

The replay changes only LN relative to `build_attention_boundary/carry/full`.
All3564 dispatches finish. Token feedback remains248045 ->8678 ->198 ->2;
all reset, state/cache and canary checks pass. Minimum full-logit correlation
is0.999984317183, above0.9999 but slightly below the preceding variant.

Numerical acceptance remains **FAIL**:18891/18896 slice and4121/4122 decode
checks pass, leaving six failures. The failed fields differ from the previous
six: final residual/norm maxrel improve, while four layer47 heads fail. Both
result files retain `passed: false` and compare exits1. Do not promote the
experimental kernel or claim PR4 complete; numerical closure, runtime
integration and performance evaluation remain pending.

| Frame | Field | maxrel | Required maxrel | Cosine |
|---|---|---:|---:|---:|
|cold-1-layer47|head1 og|0.021276595745|<0.02|0.999817890284|
|cold-1-layer47|head11 og|0.024875621891|<0.02|0.999907686740|
|cold-1-layer47|head17 og|0.023109243697|<0.02|0.999861453771|
|cold-1-layer47|head22 og|0.026905829596|<0.02|0.999900004724|
|cold-1-layer63|y|0.020429167152|<0.005|0.999985680192|
|cold-1|final norm|0.012820512821|<0.008|0.999984037074|

Attention heads also require cosine>0.9999. Final residual maxrel was
0.028256613553 and final norm0.017094017094 in the preceding variant.

All384 residual additions remain exact (0/1966080 differing values). Local norm
differences drop from26 to9; total model norm differences drop from827992 to
809914. These diagnostic counts overlap with propagated error and do not
replace acceptance. The first ten norm boundaries, through layer4 post norm,
are exact. The first changed norm is layer5 xn:five propagated differences,
zero local. The first local norm difference is now layer7 post norm.

## Next isolated target

`diagnose-wide-ffn.py --out <full> --tag cold-0-layer4` verifies exact xm but
finds one BF16 activation mismatch at index14949. Device h is
0.0018959046574309468 versus independent FP32 h0.001895904541015625. Rounding
to BF16 gives0.00189971923828125 versus0.00189208984375. The tiny FP32 error
therefore changes the down-projection input.

The layer4 FFN output maximum absolute error is3.0547380447e-7; conditional
down projection from the actual device activation has only3.7252902985e-9.
This identifies an activation rounding boundary for further isolation, not a
proof that every remaining full-model failure has the same cause. Next replay
the up/gate traces to separate projection from SiLU/product rounding at14949.
Keep the original model oracle. Logs: `/tmp/norm-full-{compare,rounding}.log`
and `/tmp/norm-layer4-ffn.log`; machine-readable diagnostics are in `full`.
