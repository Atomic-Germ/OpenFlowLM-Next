# Real banked AB projection precision

2026-10-03. Continue from the four numerical failures in
[DeltaNet projection carry](qwen38-deltanet-boundary.md). That experiment
localized the remaining layer10/head4 gated-output rounding to AB.

## Reproduction and TDD

`utilities/diagnose-wide-ab.py` replays real xn and packed AB constants from the
full capture using the production byte-copy adapters. It validates hashes,
keeps guard bytes and output poison, and executes two independent repetitions.
The baseline exactly reproduces recorded AB. A strict all-head dot comparison
fails:47/48 alpha and47/48 beta-logits differ from FP64 projection rounded to
FP32; maxabs7.3909759521484375e-6 and1.1444091796875e-5.

At head4, baseline beta-logit is1.9046249389648438 instead of1.9046193361282349.
Its sigmoid0.8704140782356262 is exact relative to that captured logit, showing
that correcting the dot is the first intervention. Baseline decay also has
local nonlinear error and is not claimed exact.

CPU tests first fail for the missing diagnostic and AB replay option. Tests
then cover conditional nonlinear math, strict dot failure, guarded repeat,
fixture tampering, kernel geometry, primitive failure and changed binary hashes.
The full CPU suite passes863 tests,47 skipped. Neural operators stay on NPU;
the independent CPU oracle is diagnostic only.

## Opt-in compensation

WIDE_AB_CARRY=1 selects a separate BF16 dot entry point using the existing
Q4 product/TwoSum helper. BF16 products are exact FP32 values in the normal
model range. The low component survives across all64-row tiles and xn chunks.
Each private32-lane accumulator becomes64 floats (high/low). Both reset on the
first tile of every bank/projection; the store folds them before applying the
unchanged NPU scalar nonlinearities. Weight packing, head order and external
BO layout are unchanged. Default remains off.

| Check | Result |
|---|---|
|H5120 primitive|28/28 checks, seven inputs|
|H2560 primitive|28/28 checks, seven inputs|
|Rebuilt default H5120|28/28; all seven outputs byte-identical to old kernel|
|Real layer10 alpha/beta-logits|All96 FP32 outputs exact; both runs identical|
|Real layer10 decay|35 FP32 differences, maxabs3.5762786865234375e-7|
|Real layer10 beta|13 FP32 differences, maxabs5.960464477539063e-8|
|Text H5120/H2560|14800/11392 B, within16384 B|
|Placed data plus reserved stack|21760 B versus21504 B baseline|

Head4 candidate beta matches the independent model value0.8704134821891785.
Its decay0.8549832105636597 still differs from0.854983389377594. The diagnostic
reports local nonlinear errors separately and never sets model_passed=true.
Matching rounded dot outputs does not imply matching the oracle's unrounded
FP64 dots inside nonlinearities. The observed warm repeated AB call is1.422 ms
versus0.560 ms baseline in this short run; compensation has a measurable cost.

| Artifact | SHA256 |
|---|---|
|H5120 xclbin|`4527fed1775d962e2e744b5a39ab037fee52066f2ba3e71d821007c1580add89`|
|H5120 instructions|`e6c4d40aa6bd58864ac7b001e9c4b810ca1f657d09102962523c83200966481b`|
|H2560 xclbin|`aba77cb1829ff5d83cb2fe1d26e0b92992d70aacc5ae8c0395230cbe296c959b`|
|H2560 instructions|`adc026547e2a9e8097117f236853c5f617ff84b9019675b4294bd1891d1059f3`|

Artifacts: `open_kernels/designs/wide_deltanet/build_ab_precision`.
Reproduction: [open-wide-ab-carry](../../../.opencode/skill/open-wide-ab-carry/SKILL.md).
Logs: `/tmp/ab-precision-*.log`.

## Remaining layer10 boundary

With the candidate AB, conditional post math on captured O/Z now agrees with
the independent BF16 output at channel604 (head4/lane92). All recorded
counterfactuals, including device AB alone, likewise give the correct rounding.
The device still produces the adjacent BF16 value, so this channel is now a
local post error rather than a propagated AB error. Its three downstream xm
differences at2315,2508,2814 persist, with zero local RMSNorm error.

- Captured O: -2.351502644160064e-6; Z: -0.29373395442962646.
- Conditional FP64 post:0.00027370451186742463.
- Conditional FP32 post:0.0002737044997047633.
- Conditional/reference BF16:0.0002727508544921875.
- Device BF16:0.000274658203125.

Do not report the entire layer10 boundary closed because its dot products are
now exact. The next isolated target is the post RMSNorm/gate arithmetic at604.
The read-only `full/layer10-post604-summary.json` and
`full/cold-0-layer10-deltanet-boundary.json` retain the evidence.

## Full-model acceptance remains open

All3564 NPU dispatches finish. Tokens248045 ->8678 ->198 ->2, reset, state
isolation, cache preservation and canaries pass. All384 residual additions
remain exact. Slice18893/18896 and decode4121/4122 pass: **four numerical
failures**, the same count as before, with a changed set of failing captures.

| Capture / field | Normalized max error | Cosine | Required bound |
|---|---:|---:|---|
|cold-1-layer47/head1_og|0.015957446808510637|0.9998718232783695|maxrel<0.02, cosine>0.9999|
|cold-1-layer47/head22_og|0.02242152466367713|0.9999117632159779|maxrel<0.02, cosine>0.9999|
|cold-1-layer63/y|0.01929894806260552|0.9999890809083201|maxrel<0.005|
|cold-1/final_norm|0.008547008547008548|0.9999876742750556|maxrel<0.008|

The preceding layer47/head11 and layer61/y failures disappear, while heads1
and22 fail now. Final residual maxrel decreases from0.020431852984951898;
final norm decreases from0.01282051282051282. Neither meets its bound. Logit
correlations are0.9999976872078569,0.9999877940504203,0.9999961102989076.

Norm counts fall to798117 model differences (from804385), 8 local (from10)
and798115 propagated (from804384), with overlap. The first21 norm boundaries
remain exact; the first model divergence is still layer10/xm, and the first
local norm error is now layer51/xn. The full comparator correctly exits1.
No thresholds, reference tensors or weights changed. The mode remains opt-in;
PR4 and runtime/catalogue promotion remain incomplete.
