# Segmented down precision

2026-10-01. Follow-up to [real FFN boundaries](qwen38-ffn-boundary.md).
The stage targets the first propagated BF16 difference at cold0/layer1,
channel2931, traced to the preceding layer's segmented down projection.

## Diagnosis and failing test

The captured layer0 h has no BF16 differences against the independent oracle.
Its down output at2931 is-0.0063009862788021564 versus-0.0063009848818182945.
Independent FP64 evaluation of the original packed pool separates the five
K4096+4096+4096+4096+1024 partials:

| Segment | FP64 partial at2931 |
|---|---:|
|0|-0.012092165934745935|
|1|0.001527444604445094|
|2|-0.010802608004496506|
|3|0.01699322337316289|
|4|-0.001926878977656088|

Summing unrounded partials and then rounding gives the reference. Rounding
each partial first, even with an FP64 final sum, gives-0.006300985347479582.
Sequential FP32 addition gives-0.0063009862788021564, exactly the device result.
Both the partial interface and the outer reduction therefore lose precision.

A CPU regression checks these partials independently against dense FP64 math,
including an unaligned final segment. A cancellation case demonstrates why
compensating already-rounded partials cannot recover the lost bits. A strict
real-frame NPU test fails before the arithmetic change, with h matching and
both repeated frames and recorded output matching exactly.

## Change

`--segment-carry` requires a full FFN probe with block/product correction.
The Q4 epilogue preserves the residual of its final FP32 rounding in the
existing per-partition table scratch. Up/gate keep their original FP32
interface. Down reads the residual before the next band overwrites it,
restores even/odd lanes to row order, and adds both components to a normalized
two-component sum. The low plane occupies the previously unused second640
floats of ds; both planes reset at the first segment of every frame.
The final output rounds high+low to the original FP32 interface.

No new buffer, pool representation, weight conversion, neural CPU fallback or
reference change is introduced. Insufficient scratch or missing prerequisites
are rejected. Other modes keep their arithmetic and remain the defaults.

## Local acceptance

| Check | Result |
|---|---|
|Real layer0 channel2931|exact, strict test passes|
|Real layer0 h|unchanged,0 BF16 mismatches|
|Real down differing FP32 values|2607 ->4 of5120|
|Real down maximum absolute error|2.3841857910e-7 ->9.3132257462e-10|
|Repeated real frame|bit-exact|
|Synthetic normal FFN|26/26 pass|
|Synthetic traced FFN|52/52 pass|
|Normal/trace h and fo|bit-exact for all13 inputs|
|Rebuilt previous mode|all13 complete captures bit-exact to the preceding build|
|CPU suite|821 passed,47 skipped|

The remaining four down differences are recorded; this is not a claim of
bit-exact down arithmetic across all channels. Synthetic worst maxrel remains
1.3845709505e-6, below the unchanged1e-4 bound.

| Probe | Data + reserved stack/core | Text/core |
|---|---:|---:|
|Normal FFN|63552 B|16240 B|
|Traced FFN|64064 B|16384 B|

The traced program reaches the16KiB program budget exactly. The actual table
is22592 B, ds remains1280 floats, and reserved stack remains6144 B. Inspect
linker placements before adding code to this variant.

## Reproduction

Use [open-down-segment-carry](../../../.opencode/skill/open-down-segment-carry/SKILL.md).
Artifacts are under `open_kernels/designs/wide_deltanet/build_down_precision`:
`baseline_layer0` retains the strict failing test; `carry/layer0_trace` the
passing test; `carry/ffn` and `carry/ffn_trace` the new kernels;
`regression/ffn` the rebuilt prior mode.

The diagnostic utility's `--down-segments --down-channel 2931` reports the
FP64 partials and reduction variants. `--compare-trace
--require-exact-h-bf16 --require-exact-fo-channel 2931` enforces the real-frame
gate and exact repeat. Conditional diagnostic references remain separate
from the full-model acceptance fixtures.

| Artifact | SHA256 |
|---|---|
|Normal FFN xclbin|`65df21d39c3bec8d46a3ab5b26d7bc7484c7a2ba33818f672bfe21b8800b7b65`|
|Normal FFN instructions|`e8cc08c29bbca5afbdfe59493baf6d4655e6853f0a4c369b25847e91337ce708`|
|Traced FFN xclbin|`df539b001f49d458f207c219332c5610ec00a57ad3c0c1e4513b711f5b0679dd`|
|Traced FFN instructions|`ee453a7f6fe43ddabd6d25ddd3bf654781171ec18acf377a11b8c968ec615775`|

Instructions are byte-identical to the previous mode for both variants.

## Full-model result and next step

`carry/full` replaces only FFN relative to `build_ffn_boundary/carry/full`.
The output projection and compensated LN are reused; all other kernels,
packed fixtures, independent references, seed tokens and tolerances are
unchanged. All3564 NPU calls complete. Three autoregressive tokens and full
reset replay choose248045 ->8678 ->198 ->2. Reset, state/cache copies,
canaries and token feedback pass. Minimum logits correlation is0.9999853336
against0.9999.

Full acceptance remains **FAIL**:18893/18896 slice checks and4121/4122 decode
checks pass. PR4 remains incomplete; runtime integration, benchmark and
catalogue promotion are still pending.

| Position | Field | Observed | Required |
|---|---|---:|---:|
|cold1/layer61|y maxrel|0.005244998555|<0.005|
|cold1/layer63|y maxrel|0.03343345291|<0.005|
|cold2/layer51|head5_og cosine|0.9998908622|>=0.9999|
|cold1/final|final_norm maxrel|0.01709401709|<0.008|

The head's maxrel0.007281553398 passes its0.02 bound; cosine fails. The new
local fix reduces five failures to four, but does not improve every global
metric: layer63 y and final norm maxrel are worse. It remains experimental.

Both norms in cold0/layers0,1,2 now match the independent reference exactly.
The first changed boundary moves to layer3 entry: channel3390, one propagated
BF16 difference and no local norm error. Layer3 post norm has72 differences.
Across384 boundaries,838592 values differ, with19 local norm errors and838593
propagated-input differences; these diagnostics overlap and are not additive.

The next isolated target is the layer1 residual addition at3390:

| Tensor | Device | Reference |
|---|---:|---:|
|layer1 res1|-0.03440730646252632|-0.03440730646252632|
|layer1 fo|0.005703141447156668|0.005703141447156668|
|layer1 y|-0.028704166412353516|-0.028704164549708366|
|layer2 y|0.00010345038026571274|0.00010345224291086197|
|layer3 xn|0.00011539459228515625|0.00011587142944335938|

Both CPU FP32 addition and FP64 addition rounded to FP32 reproduce the
reference layer1 y from the two matching captured inputs. Layer2 projout and
fo are also exact at this channel, so the-1.8626451492e-9 residual difference
persists into its y and flips layer3 entry rounding. Isolate the LN residual
add path with these same inputs before changing its arithmetic or references.
