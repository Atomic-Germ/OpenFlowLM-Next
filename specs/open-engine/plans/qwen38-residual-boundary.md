# Residual boundary at layer11/channel4872

2026-10-04. Continue from [post product carry](qwen38-post-precision.md).
The baseline is `wide_deltanet/build_post_precision/full`, whose first23 norm
boundaries agree with the immutable independent reference.

## Counterfactual diagnosis

`utilities/diagnose-wide-residual.py` checks kernel/reference hashes and capture
sizes/canaries, then compares all-device, all-reference, device-input-only and
device-projection-only residuals. Single-channel substitutions distinguish a
local numerator error from changes in the global RMS denominator. All residual
sums round to FP32 before RMSNorm; results are always diagnostic_only and never
claim model acceptance. Padded BF16 norm captures exclude poison lanes from
finite checks. The tool also records the scalar error history for both residual
additions in each preceding layer of the same token.

At cold-0-layer11/xm[4872]:

|Residual operands|BF16 value|Differences from independent norm|
|---|---:|---:|
|Device input and projection|-0.0034332275390625|1|
|Reference input and projection|-0.003448486328125|0|
|Device input, reference projection|-0.0034332275390625|1|
|Reference input, device projection|-0.003448486328125|0|
|Only input channel4872 from device|-0.0034332275390625|1|
|Device with input channel4872 repaired|-0.003448486328125|0|

Conditional RMSNorm is exact. The error therefore arrives in the residual
input at this frame; the four tiny layer11 output-projection differences do
not explain the BF16 mismatch. These are conditional substitutions, not a
replacement execution of the model.

The first scalar difference is layer0 FFN down output4872:
0.010909566655755043 versus0.010909565724432468, one FP32 step
(9.313225746154785e-10). Its correctly rounded residual sum differs by
1.862645149230957e-9. Subsequent residual rounding increases that to
1.4901161193847656e-8 by layer7. A small layer10 projection error reduces it
to1.30385160446167e-8, which reaches layer11's BF16 boundary. All recorded
scalar residual additions themselves match IEEE FP32.

Layer0's xm and all17408 BF16 FFN activations are exact. Conditional down and
independent down comparisons both find four FP32 output differences, with
maxabs9.313225746154785e-10. The original traced FFN reproduces the recorded
activation and output; strict `--require-exact-fo-channel 4872` exits1.

## TDD and experiments

CPU tests first fail for the missing diagnostic and build option, then cover
operand substitution, individual-channel substitution, FP32 residual rounding,
read-only inputs, poison padding, guard/nonfinite rejection, reference tampering
and prerequisite validation before any build output is created.

A first experiment replaces only the final high+low addition with the existing
integer-lane IEEE adder. It passes all52 synthetic trace checks and reduces
layer0 down differences from four to three, but4872 remains wrong. Artifacts
are retained in `build_residual_precision/final_rne` and `final_rne_layer0`.
Its traced text is15552 B. This experiment does not close the target gate.

Further experiments applied TwoSum also to segment accumulation, retaining
both high and low planes and their per-frame resets, but exceeded program
memory and are not retained in source. The retained opt-in `--down-rne` changes
only final rounding and requires `--segment-carry`; packed weights, buffers and
DMA schedule are unchanged. Default arithmetic remains untouched. Neural math
remains on NPU. It fixes channel2743;4872 is explicitly still open.

Artifacts and logs: `open_kernels/designs/wide_deltanet/build_residual_precision`,
`/tmp/residual-*.log`, `/tmp/down-rne-*.log`. Original references, tolerances,
model weights and the accepted preceding binaries remain immutable.

The unsorted exact reducer overflowed program memory. A magnitude-ordered
FastTwoSum variant with all four additions/subtractions using the integer
adder still occupied16800 B, exceeding16384 B. The compact variant retains
exact rounded additions and uses native subtraction for the two error-recovery
operations after ordering by magnitude. Those differences are representable
in the finite normal range. Experimental CPU checks covered10000 pairs over exponents-50..50,
including cancellation, small tails and zero; hardware acceptance is separate.

The native-subtraction variant still measured16624 B; minsize on its helper
made no difference. Failed artifacts are retained under sorted_rne, compact_rne
and size_rne for instruction-size diagnosis. No numerical result is claimed
for kernels that did not produce a loadable xclbin. The final retained source
uses the first, successfully built final-addition change only.

## Retained implementation

The five new CPU tests pass; full suite **873 passed,47 skipped**. The generic
sorted-sum experiment and its test are not included as unused/unbuildable code.
The retained final adder reuses `act_add` from the existing compensated activation
header, so it adds only24 bytes of text. Both compile-time and probe CLI switches
default off. The builder records down_rne in its provenance and source cache key.

|Artifact|Text|SHA256 xclbin|SHA256 instructions|
|---|---:|---|---|
|Normal|15424 B|`422c719ce3041bbe0aaa21bba6e8ccc0bd986e72a4664cf9fca6abf69a202971`|`e8cc08c29bbca5afbdfe59493baf6d4655e6853f0a4c369b25847e91337ce708`|
|Trace|15552 B|`a21f409d4c42f13e61a98efe2d95526010677d562f5d329a69b90bcefbdc3788`|`ee453a7f6fe43ddabd6d25ddd3bf654781171ec18acf377a11b8c968ec615775`|

Real channel2743 changes from -4.38264896729379e-6 to the independent
-4.382648512546439e-6. Prior corrected channel2931 remains exact; all BF16 h
values and repeated captures remain exact. Strict4872 still fails, with channels
392 and1769 also retaining tiny down errors. This is partial progress, not a
closed layer11 boundary or PR4 acceptance.

Normal/trace primitive acceptance passes26/52 checks, and all13 activation
arenas match bytewise between the two builds. Rebuilt default mode passes26
checks and matches all13 preceding `build_main_oct3/ffn` arenas bytewise.
Actual ELF .text is15424/15552 B; the separately placed40-byte coefficient
data makes plain llvm-size's combined column40 B larger. Use llvm-size -A.
Placed data including the reserved6144-byte stack is63592/64104 B.

The new full replay is `build_residual_precision/full_down_rne`. The existing
`build_residual_precision/full` belongs to an earlier residual-addition stage;
the fresh-directory guard rejected that collision before writing anything.

Next isolate down-only segment outputs and their low components at4872, with
the captured exact BF16 h. A down-only probe can test the larger reducer without
linking the full SiLU path. Establish where the remaining error enters before
spending the full FFN's instruction budget on compensation; the unexecuted
segment variants do not prove that segment addition is the sole cause.

Reproduction: [open-down-rne](../../../.opencode/skill/open-down-rne/SKILL.md).

## Full replay

All3564 NPU dispatches finish. All384 captured xn/xm buffers are byte-identical
to `build_post_precision/full`, and rounding-diagnosis.json is identical:
805787 model differences,4 local norm differences and805786 propagated
differences, with overlap. All384 residual additions remain exact. The first23
norm boundaries match the model; layer11/xm[4872] retains the same discrepancy
and operand-substitution results. The isolated layer0 FFN candidate agrees
exactly with its full-model capture. No closed full-depth boundary is claimed
from fixing channel2743.

Slice18892/18896 and all4122/4122 decode checks pass. The combined comparator
correctly exits1: decode-results.json retains passed=false because slice
acceptance fails. Its contents are identical to the preceding stage. Tokens
remain248045 ->8678 ->198 ->2, with unchanged logit correlations
0.9999980214079868,0.9999882250097705,0.9999954102089279. Reset, state/cache
isolation and canaries pass.

|Remaining capture / field|Maxrel|Cosine|Required bound|
|---|---:|---:|---|
|cold-1-layer47/head1_og|0.02127659574468085|0.9998099379797555|maxrel<0.02, cosine>0.9999|
|cold-1-layer55/y|0.005122083271934342|0.999992395071764|maxrel<0.005|
|cold-1-layer63/y|0.010158542642347071|0.9999901219936939|maxrel<0.005|
|cold-2-layer39/head4_og|0.017391304347826087|0.9998821524480384|maxrel<0.02, cosine>0.9999|

The failing set and max errors are unchanged. Tiny FP32 improvements do not
close a model gate in this run. PR4/runtime/catalogue promotion remain incomplete;
keep the mode opt-in and proceed with the isolated down-segment diagnosis.

## Follow-up: segment trace

[The down-only replay](qwen38-down-segment-trace.md) now reproduces the complete
FFN output bytewise. At channel4872 the error is already inside segment3,
K12288..16383; reduction of captured high/low segments is exact. Channels392
and1769 likewise have intra-segment errors. The next target is Q4 block
accumulation within those segments, rather than a larger segment reducer.
