# Real FFN activation rounding

2026-10-02. Follow-up to [RMSNorm precision](qwen38-norm-precision.md).
This stage isolates the layer4 activation mismatch and adds opt-in compensated
SiLU/product arithmetic without changing packed weights or model references.

## Failing evidence and TDD

The real cold0/layer4 frame has exact BF16 xm. Tracing the previous FFN shows
up0.12876082956790924 and gate0.02902720682322979 at index14949, both exactly
their independent FP32 references. Device h is0.0018959046574309468 versus
reference0.001895904541015625. This changes BF16 from0.00189208984375 to
0.00189971923828125. The strict real-frame gate fails:one BF16 error from
activation, zero from projections. Trace and source captures match bitwise.

The CLI confinement test initially failed for the absent activation mode.
The existing strict real-frame test supplies the numerical RED gate. The
compact-loop worker tests exercise actual segment order, per-band accumulation
and reset on a second token, with and without segment residual carry. Existing
integer-adder tests validate the reused exact-add algorithm. CPU result:
**835 passed,47 skipped**.

## Implementation and program-memory constraint

`--activation-carry` requires full FFN product correction. Its separately named
helpers split FP32 factors into three BF16 components, retain all nine products
and carry accumulation residuals using the existing Q4 product helper. The
polynomial and reciprocal Newton steps use the previous coefficients and two
iterations, with exact FP32 add/sub via the existing integer-lane adder.
Products remain a compensated BF16 implementation; this is not globally exact
FP32 or FP64 SiLU. Other operators and default arithmetic remain unchanged.

Several measured constraints determine the final implementation:

- All-IEEE integer activation arithmetic exceeded program memory, even after
  a size-oriented compilation experiment. Those failed modes are not shipped.
- Compensated multiplication with unrolled Horner evaluation occupied16912 B
  of program text,528 B too much. Retaining the failed ELF made this measurable.
- Rolling the six Horner stages kept coefficients and evaluation order intact
  and reduced text to16368 B plus24 B coefficient data. Its primitive gates
  passed, but14949 still failed:product compensation alone was insufficient.
- Exact additions need more text. The opt-in core schedule now loops over
  consecutive equal-width down segments. The four K4096 segments execute the
  same body with first-segment reset computed in the loop, then the K1024 tail.
  Host segment-major DMA remains byte-identical. This makes room for exact add.

`utilities/build-design-retain-failure.py` is a diagnostic wrapper for the
installed mlir-aie cleanup hook; it preserves ELF files and the failing exit
code. It does not alter the normal build tool or installed package.

## Primitive, real-frame and compatibility results

| Check | Result |
|---|---|
|Normal FFN|26/26 checks pass over13 inputs|
|Traced FFN|52/52 checks pass over13 inputs|
|Normal versus trace|All13 captured activation arenas bit-exact|
|Layer4 h BF16|17408/17408 exact, repeated run exact|
|Earlier layers0 and1 h BF16|Exact, repeated runs exact|
|Layer0 down channel2931|Still exact|
|Rebuilt previous mode|26/26 pass; all13 arenas bit-exact to prior build|
|CPU suite|835 passed,47 skipped|

Layer4 up/gate remain unchanged:8 and5 FP32 differences respectively across
their complete tensors, none affecting BF16 h. Local activation h maximum
absolute error rises from1.4901161194e-8 to2.3841857910e-7, although the sole
BF16 mismatch closes. This is a rounding-boundary fix, not a uniform reduction
of every FP32 error. FFN output max absolute error falls from3.0547380447e-7 to
7.4505805969e-9 (about41 times); no reference or threshold is changed.

| Resource | Normal | Trace |
|---|---:|---:|
|Program text/core|16176 B|16320 B|
|Coefficient data/core|24 B|24 B|
|Allocated buffers/data plus reserved stack/core|63576 B|64088 B|

The normal DMA instructions match the previous FFN byte-for-byte. A short
13-input sample measured median dispatch13.266 ms versus12.498 ms for the
previous mode; this is not a stable application benchmark.

| Artifact | SHA256 |
|---|---|
|Normal xclbin|`8774af5c84d022db5420bc428d46c4e46f4290bf42b30991384a646bd94b7d48`|
|Normal instructions|`e8cc08c29bbca5afbdfe59493baf6d4655e6853f0a4c369b25847e91337ce708`|
|Trace xclbin|`dc0c247bf3ef5578ad63cb870de736e71e929973dcaa532f4cdad8b4c2242d4c`|
|Trace instructions|`ee453a7f6fe43ddabd6d25ddd3bf654781171ec18acf377a11b8c968ec615775`|

## Reproduction

Use [open-ffn-activation-carry](../../../.opencode/skill/open-ffn-activation-carry/SKILL.md).
Artifacts are under `open_kernels/designs/wide_deltanet/build_activation_boundary`.
`baseline` is the failing real-frame trace. `rne`, `rne_compact` and `carry`
record rejected program-memory experiments; the retained `carry` ELF is16912 B.
`carry_loop` fits and passes primitives but fails the real rounding gate.
`carry_add` is the final implementation, with normal/trace kernels, real frames
for layers0/1/4, and the full replay. `regression/ffn` retains the old-mode rebuild.
Relevant logs use `/tmp/activation-*`; final hardware output lives in
`carry_add/full/hardware.log`. Preserve accepted fixture directories.

## Full-model replay

Only FFN changes relative to `build_norm_boundary/full`. The original full
fixtures, model, tokens, reference tensors and thresholds are unchanged.
All3564 NPU calls complete; feedback is248045 ->8678 ->198 ->2. Reset replay,
state/cache preservation and canaries pass. Minimum full-logit correlation is
0.999985863109, above0.9999.

Numerical acceptance is still **FAIL**:18895/18896 slice checks and4121/4122
decode checks pass. Two failures remain versus six previously, but both final
errors increase. No attention-head checks fail in this run. Both result files
retain `passed: false`; the full compare exits1. Keep the mode experimental:
this is not accepted runtime support or completion of PR4.

| Frame/field | Previous maxrel | New maxrel | Required bound |
|---|---:|---:|---:|
|cold-1-layer63 y|0.020429167152|0.038299458784|<0.005|
|cold-1 final norm|0.012820512821|0.021367521368|<0.008|

All384 residual additions remain exact (0/1966080 differing values). The first
ten norm boundaries still match exactly. The first changed norm remains
layer5 xn, but its five propagated errors have become one local norm error
with zero propagated errors. Across384 boundaries,826145 values differ from
the independent path and16 differ from conditional norms, versus809914 and9
before. These trajectory-dependent diagnostics overlap and do not replace
the numerical gates.

## Next isolated target

The layer5 input norm differs at channel3295: input0.028717929497361183,
weight1.078125, device0.0252685546875, conditional and model0.025390625.
A separate guarded replay of the unchanged norm kernel reproduces exactly
this one difference, with exact residual and repeated outputs. This is the
next local RMSNorm rounding target; it does not imply that every final-layer
failure has that cause.

Artifacts: `carry_add/full/{slice-results,decode-results,rounding-diagnosis}.json`
and `carry_add/layer5_norm/replay-results.json`. The latter intentionally fails
the strict local gate and is a reproducible starting point for the next stage.
Logs: `/tmp/activation-full-{prepare,compare,rounding}.log` and
`/tmp/activation-layer5-norm-{hardware,compare}.log`.
