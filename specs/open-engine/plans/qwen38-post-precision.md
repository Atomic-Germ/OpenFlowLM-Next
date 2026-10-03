# DeltaNet post RMSNorm/gate precision

2026-10-03. Continue from [banked AB compensation](qwen38-ab-precision.md).
The next isolated error is cold-0-layer10/channel604 (head4/lane92), after the
AB dot products and Z projection have been corrected.

## Reproduction and TDD

The new `utilities/test-wide-post.py` replays captured O/Z and BF16 norm weights
with guarded, repeated NPU calls. It also supplies independent synthetic
acceptance for 32/48 heads. Baseline real replay reproduces exactly one BF16
difference at604; `--require-exact` correctly fails. The FP64 conditional value
is0.00027370451186742463, which rounds through FP32 to BF16
0.0002727508544921875. Baseline produces0.000274658203125.

POST_TRACE adds six FP32 planes without changing baseline output. The trace
shows correctly rounded sum-of-squares and inverse root at the target, followed
by one-ULP errors in normalized O and SiLU. The final FP32 result lands exactly
on the BF16 midpoint0.00027370452880859375. Correctly rounding each intermediate
product alone still reaches that midpoint: retaining product residuals across
the whole expression is necessary for this experiment.

|Channel604 trace|Baseline|Candidate|
|---|---:|---:|
|sum_sq|5.162521574675338e-7|same|
|inv|997.9894409179688|same|
|normalized|-0.0023467745631933212|-0.002346774796023965|
|weighted|-0.0021817670203745365|-0.00218176725320518|
|silu|-0.12545084953308105|-0.12545083463191986|
|result|0.00027370452880859375|0.0002737044997047633|

CPU tests fail first for missing replay, trace reconstruction, compensated
product and full-model override support. They then verify per-head FP64
reference, strict BF16 failure, repeats, canaries, hash tampering, trace/output
consistency, 10000 five-factor products against FP64 and full-model binding
rejection for traced, conditional, wrong-geometry or failed primitive fixtures.
Full CPU suite: **868 passed, 47 skipped**.

## Opt-in kernel and hardware checks

POST_CARRY=1 preserves low product residuals across O * inv * weight * Z *
sigmoid(Z), using existing exact integer-lane FP32 operations and the small-Z
sigmoid series. Outside abs(Z)<=0.5 it retains precise exp/reciprocal. Norm
sum and scalar inverse root are unchanged. Default is off; four-BO ABI and
legacy fused arithmetic are unchanged. Neural math stays on the NPU.

This is compensation for finite normal-range inputs, not general IEEE
overflow/underflow handling or an exact sigmoid guarantee. In carry traces,
normalized/weighted/SiLU are rounded diagnostic previews, not intermediate
inputs to the final compensated product.

| Check | Result |
|---|---|
|Real layer10, normal and traced candidate|All6144 BF16 values exact; two identical runs each|
|Target final FP32|0.0002737044997047633, matching conditional reference|
|Synthetic 48/32 heads|All three cases BF16-exact, repeats/guards pass|
|Trace versus normal|Byte-identical on synthetic cases and real layer10|
|32/48 shared prefix|Byte-identical on all three cases|
|Rebuilt default versus prior post|Byte-identical on all three cases|
|Text 48/32/48-trace|14880/14512/15440 B, each below16384 B|
|Placed data plus reserved stack|22784 B normal;47360 B traced, each below65536 B|

Stack reservation is6144 B; traced FIFO adds24576 B. The measured repeated
real normal call takes4.596 ms versus0.236 ms baseline, a substantial precision
cost. These short isolated calls are not an application benchmark.

| Artifact | SHA256 |
|---|---|
|48-head xclbin|`d51c03496123fcc5e2291499a27ce14e613048f1760eb6f137488104fe975423`|
|48-head instructions|`3026b00196bd251c7e0a1852ffae17eff94e44604ea23c8d9e3d3cd1783907cb`|
|32-head xclbin|`6a5b1736c3847a4f9507684817ea83d8bb856468a4313b69a77fb4d9c88df634`|
|32-head instructions|`3139c8cdff0d68452efbd7d52c12e0251902e0b5cacd40c46bc7e2f1bca8db1b`|

Artifacts: `open_kernels/designs/wide_deltanet/build_post_precision`.
Workflow: [open-wide-post-carry](../../../.opencode/skill/open-wide-post-carry/SKILL.md).
Logs: `/tmp/post-precision-*.log`. No thresholds, weights or model reference
tensors are regenerated. PR4 completion depends on the separate full-model gate.

## Boundary movement in the full model

All3564 dispatches finish. At layer10, all gated BF16 output and post-norm xm
now match the independent model path, closing the previous three xm differences.
The first23 norm boundaries are exact (previously21). The first model difference
is cold-0-layer11/xm[4872]: device -0.0034332275390625 versus reference
-0.003448486328125. Conditional RMSNorm on device residual is exact, as is the
layer11 gated attention output; this is propagated residual error. Four output
projection FP32 channels differ, and457 residual channels carry differences.
Do not infer that the projection alone causes xm[4872]: earlier residual errors
also survive despite identical BF16 norm outputs.

Across all384 norm boundaries, model differences increase from798117 to805787,
local norm differences decrease from8 to4, and propagated differences increase
from798115 to805786 (categories overlap). All384 residual additions remain
FP32-exact. The first local norm discrepancy is cold-1-layer14/xn. Closing the
isolated layer10 boundary does not establish global improvement at every layer.

Next isolate layer11/xm[4872] with input-residual/output-projection
counterfactuals, preserving the original weights, references and tolerances.
The captured `layer11-boundary-summary.json`, `cold-0-layer11-diagnosis.json`
and `rounding-diagnosis.json` retain the evidence.

## Full-model gate remains open

Slice18892/18896 checks pass. All4122/4122 decode checks pass, including the
previous failing final norm: cold-1 maxrel falls from0.008547008547008548 to
0.004607371794871795, below0.008. The combined comparator still exits1 and
decode-results retains passed=false because slice acceptance fails.

|Remaining capture / field|Maxrel|Cosine|Required bound|
|---|---:|---:|---|
|cold-1-layer47/head1_og|0.02127659574468085|0.9998099379797555|maxrel<0.02, cosine>0.9999|
|cold-1-layer55/y|0.005122083271934342|0.9999923950717399|maxrel<0.005|
|cold-1-layer63/y|0.010158542642347071|0.9999901219936939|maxrel<0.005|
|cold-2-layer39/head4_og|0.017391304347826087|0.9998821524480384|maxrel<0.02, cosine>0.9999|

The total remains four failures. Layer47/head22 and final_norm now pass;
layer55 residual and cold-2/layer39 head4 now fail. Final layer63 residual
improves from0.01929894806260552 but still exceeds0.005. Do not describe this
as uniform improvement: layer47/head1 worsens and overall norm differences grow.

Tokens remain248045 ->8678 ->198 ->2. Logit correlations are
0.9999980214079868,0.9999882250097705,0.9999954102089279. Reset, recurrent state
isolation, KV preservation and all canaries pass. Defaults remain unchanged;
runtime/catalogue promotion and PR4 remain incomplete.
