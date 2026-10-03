# Real small-argument sigmoid precision

Follow-up to [main integration](qwen38-main-merge.md). The starting frame is
`build_norm_finish/layer8_ffn`, cold0/layer8. Its xm is exact; up/gate at749
are exact, but one activation rounds to the wrong BF16 neighbour. The strict
`--require-exact-h-bf16` replay is the RED gate for this stage.

## Isolated production trace

`layer_x/activation_probe.py` compiles the production compensated helpers in
a small diagnostic kernel. `utilities/trace-wide-activation.py` extracts the
32-lane band containing a selected channel from a hash-checked real FFN trace.
Inputs are gate32 then up32 FP32 values. Outputs are ten32-lane FP32 planes:
exp, denominator, reciprocal estimate, first/second Newton refinements, SiLU,
h, up*gate, reassociated h and a separate production-function h. Two calls
check repetition, with output poison, canaries and fixture hashes. Comparisons
always retain `diagnostic_only: true`, `passed: false` and exit1: tracing
cannot replace primitive or full-model acceptance. The first nine planes
describe the original exp/Newton path, even with the experimental series flag.

| Channel749 quantity | Device | Independent FP64 / FP32 interface |
|---|---:|---:|
|gate|-0.11944595724344254|same FP32|
|up|0.18203915655612946|same FP32|
|exp(-gate)|1.1268723011016846|1.126872343133882|
|denominator|2.1268723011016846|2.126872343133882|
|reciprocal, both refinements|0.47017398476600647|0.4701739637680983|
|h|-0.010223388671875|-0.010223387740552425 (FP32)|

The initial exponential is correctly rounded FP32 at this input, but dropping
its residual changes the required reciprocal rounding. Both Newton iterations
leave the same value. Reassociating the final products also leaves h unchanged;
this is not just a multiplication-order issue. The standalone production h
matches the full FFN's recorded32-lane band exactly.

## Candidate arithmetic

`--activation-series` requires `--activation-carry`, hence the corrected full
FFN prerequisites. For |gate|<=0.5, it evaluates the direct odd sigmoid series
through degree9, avoiding exp and reciprocal rounding in the selected value:

```
sigmoid(g) = 1/2 + g*(1/4 + g^2*(-1/48 + g^2*(1/480
                      + g^2*(-17/80640 + g^2*31/1451520))))
```

The first omitted term is at most1.1e-9 on this interval; FP32 operation error
is separate. Production uses compensated products and exact additions.
Speculative series lanes outside the interval receive zero, and the existing
exp/Newton result is selected there. The final gated products retain their
order. This is an opt-in precision experiment, not a general exact sigmoid.

Tests first fail for the absent diagnostic utility, absent series header and
missing CLI prerequisite check. The production polynomial then passes a
200001-point grid on[-0.5,0.5] plus the two real boundaries against independent
FP64 sigmoid: absolute error<4e-8, monotonic grid outputs and correct BF16 h
for749 and14949. Trace tests reject corrupted captures and changed fixtures,
and forbid treating even exact diagnostic outputs as model acceptance.

On NPU the isolated candidate produces exactly-0.010223387740552425 at749,
with zero BF16 differences across its32-lane band and exact repeat. A separate
real band containing gate1.0312272310256958 preserves the original production
value at that channel outside the series interval.

## Program-memory investigation

The first monolithic trace candidate has17264 B of text:880 B beyond the
16384 B program limit, plus40 B of coefficient data in data memory. Retained
failed ELF files make this measurable. Combining full down segments and the
shorter tail into one runtime-sized loop lowers text to16880 B, still too big.
An isolated `-Oz` experiment increases the integer-adder code and is rejected.

The selected implementation also compacts up/gate traversal and trace copies,
and outlines the unary sigmoid separately from final gated multiplication.
These transformations retain weight order, per-segment band order and reset
on the first segment of every token. CPU worker simulations cover both segment
limits and repeated tokens, with and without residual carry. Host DMA and
packed weights remain unchanged. Defaults retain their previous loops/math.

## Primitive and real-frame acceptance

| Check | Result |
|---|---|
|Normal / traced FFN primitive|26/26 and52/52 checks pass|
|Normal versus trace|All13 activation arenas byte-identical|
|Layers0,1,4,8 strict h BF16|All17408 values exact per frame; repeated runs exact|
|Layer0 down channel2931|Still exact|
|Layer8 gates outside series interval|All17 h values bit-identical to old mode|
|Previous mode rebuilt|26/26 checks pass, all13 arenas byte-identical to before|
|CPU tests|854 passed,47 skipped|
|Normal / trace text|15440 /15568 B|
|Coefficient data|40 B|
|Normal / trace allocated data plus stack|63592 /64104 B|
|Host DMA|Instructions byte-identical to previous mode|

Layer8 up/gate are unchanged. The activation's maximum FP32 absolute error
remains7.450580596923828e-9, but the sole BF16 error closes; down maxabs falls
3.56137752532959e-6 ->2.3283064365386963e-10, with only2 FP32 differences left.
This distinguishes the corrected rounding boundary from a claim that every
intermediate FP32 value became exact. Median dispatch over13 primitive calls
is20.504 ms versus20.132 ms for the rebuilt previous mode in this short run;
this is not a stable application benchmark.

| Artifact | SHA256 |
|---|---|
|Normal FFN xclbin|`1ade9eaf6c325af92bb7ffa893c17904b85e31118b78252afc0ae25496e09198`|
|Normal instructions|`e8cc08c29bbca5afbdfe59493baf6d4655e6853f0a4c369b25847e91337ce708`|
|Trace FFN xclbin|`9b8e562e391dc0449253814488ab8e16ea4df6d21c73a17163835e0582adfbea`|
|Trace instructions|`ee453a7f6fe43ddabd6d25ddd3bf654781171ec18acf377a11b8c968ec615775`|

Build and measurement details are recorded in
[open-ffn-sigmoid-series](../../../.opencode/skill/open-ffn-sigmoid-series/SKILL.md).
Artifacts are retained under `open_kernels/designs/wide_deltanet/build_activation749`;
logs use `/tmp/activation749-*.log`.

## Full-model replay: numerical gate remains open

The candidate completes all 3564 dispatches with the immutable reference
fixtures. Tokens remain 248045 -> 8678 -> 198 -> 2; reset, state isolation,
cache preservation and canaries pass. All 384 residual additions remain exact.
Nevertheless, **18891/18896 slice checks and 4121/4122 decode checks pass**:
six numerical failures versus two before this experiment. The mode remains
opt-in and is not promoted to runtime/catalogue defaults. PR4 is incomplete.

| Failing capture / field | Normalized max error | Cosine |
|---|---:|---:|
|cold-1-layer47/head1_og|0.02127659574468085|0.9997882370834705|
|cold-1-layer47/head10_og|0.013636363636363636|0.9998849841839103|
|cold-1-layer47/head11_og|0.024875621890547265|0.9999072081372572|
|cold-1-layer47/head17_og|0.023109243697478993|0.999884752964626|
|cold-1-layer63/y|0.026122822719335435|0.9999877732520025|
|cold-1/final_norm|0.01282051282051282|0.9999869882448157|

Attention-head limits remain maxrel <0.02 and cosine >0.9999; head10 fails
cosine despite passing maxrel. Final residual maxrel worsens from
0.022905298362964267 and exceeds 0.005. Final norm maxrel is unchanged and
exceeds 0.008. Full-logit correlations are 0.9999975281861762,
0.9999870100804907 and 0.999995653706738. Matching argmax does not close these
intermediate accuracy gates.

The first 21 norm boundaries are now exact, versus 18 before. Norm diagnostics
count 811003 device/reference differences (previously 810497), 9 local
differences (previously 11), and 811000 propagated differences (previously
810495); local and propagated sets overlap. The first changed norm is
`cold-0-layer10/xm`: seven propagated differences, zero local norm errors.
The first local norm error occurs at layer21/xn.

Next investigate the layer10 DeltaNet path before its post norm. Its input xn
is exact; conv capture has two BF16 differences, gated output six, and xm seven.
These comparisons against the independent trajectory do not yet establish
which projection, conv or post operation first introduces the error. The
read-only `full/layer10-boundary-summary.json` records offsets and error counts
for that investigation. References and thresholds remain unchanged.

Full artifacts: `build_activation749/full/{slice-results,decode-results}.json`
and `hardware.log`. Comparison and rounding logs are
`/tmp/activation749-full-{compare,rounding}.log`.
