# Real FFN rounding and Q4 block carry

2026-10-01. Follow-up to [Q4 product precision](qwen38-q4-product-precision.md).
This stage closes the real cold0/layer1 FFN activation rounding regression.
The new arithmetic remains opt-in; full-model acceptance is a separate gate.
**The local regression is closed; full-model acceptance still fails and PR4
is not complete.**

## Causal diagnosis before the change

The saved `build_product_precision_v2/full_ffn_ln` run has exact layer1 BF16 xm.
Its FFN output differs from the independent reference by3.2782554626e-6
maximum absolute error. Recomputing down from the device's actual h reduces
that error to1.4901161194e-8, locating the dominant error before down.

`utilities/diagnose-wide-ffn.py` reads the real packed weights and guarded
captures, checks their provenance and computes conditional FP64 references.
An isolated trace replays the unchanged real FFN input twice, exposing the
production up/gate results before activation. The original trace kernel's h
and fo match the recorded full-model captures bit-for-bit; repeat is exact.

Six h values cross BF16 boundaries, at indices410,7394,7730,8000,8902,11456.
All six are already explained by up/gate errors; activation math adds zero
BF16 mismatches on this frame. For example, gate8000 is1.9557774067e-8 on the
device versus9.7497832030e-9 from the packed-weight reference. The differing
h at410 and8902 is caused primarily by the up projection.

The strict real-frame gate (`--compare-trace --require-exact-h-bf16`) fails
before the arithmetic change. Conditional references never replace the
independent full-model trajectory or relax its tolerances.

## Change

`--block-carry` requires `--product-correction`. Instead of rounding a block's
high+low result to FP32 before the global Kahan sum, it adds both components
to the running expansion and renormalizes after every K32 block. The low part
persists in the existing table's per-partition scratch and is reset at the
first K tile. The last tile rounds the combined result for the original FP32
interface. Default and product-only modes retain their original arithmetic.

The table sizes, pool format, activation boundaries, DMA instruction layout,
stack reservation and FFN4096+4096+4096+4096+1024 segmentation are unchanged.

## Local and primitive results

| Check | Result |
|---|---|
| Real layer1 h BF16 mismatches |6 ->0|
| Real up maximum absolute error |4.7683715820e-7 ->1.4901161194e-8|
| Real gate maximum absolute error |1.1920928955e-7 ->1.4901161194e-8|
| Real up differing FP32 values |12042 ->10 of17408|
| Real gate differing FP32 values |12017 ->14 of17408|
| Real frame repeat |bit-exact|
| Synthetic full FFN |26/26 pass|
| Synthetic traced FFN |52/52 pass|
| Normal/trace h and fo |all13 inputs bit-exact|
| K6144/N5120 projection |14/14 pass, including5 exact cancellation cases|
| Rebuilt product-only projection, block carry disabled |14/14 captures bit-exact to the previous build|
| CPU suite |817 passed,47 skipped|

Worst synthetic FFN maxrel is1.3845709505e-6 (limit1e-4), versus7.1539954653e-6
in the preceding product-only mode. For the two random projection inputs,
maxrel is1.4271275343e-9 and5.4212934877e-9; the all-ones case remains
9.2438425840e-8. No input, weight, seed or acceptance tolerance was changed.

| Probe | Data + reserved stack/core | Text/core |
|---|---:|---:|
| Projection |62976 B|5712 B|
| FFN |63552 B|15760 B|
| Traced FFN |64064 B|15904 B|

These are actual linker placements. The trace arena reserves438272 bytes
versus397312 in the full-model harness; preparation preserves logical bytes,
poisons extra padding and moves the canary. Only immutable inputs are linked;
diagnostic references are copied and hashed so later diagnosis cannot change
an already prepared trace.

## Reproduction and artifacts

See [.opencode/skill/open-q4-block-carry/SKILL.md](../../../.opencode/skill/open-q4-block-carry/SKILL.md).
Builds and results are in `open_kernels/designs/wide_deltanet/build_ffn_boundary`.
The `ffn_trace` and `layer1_trace` directories retain the failing baseline;
`carry/ffn_trace`, `carry/ffn`, `carry/projection_k6144` and `carry/layer1_trace`
contain the corrected probes and strict real-frame result.

| Artifact | SHA256 |
|---|---|
| FFN xclbin | `a333a630c8546506fd7270f1791c1680fd04a22451169d11d989edcc7d6d0293` |
| FFN instructions | `e8cc08c29bbca5afbdfe59493baf6d4655e6853f0a4c369b25847e91337ce708` |
| Trace xclbin | `4e0e64123ebb42595d2216cda12512174f28f1fc16b23856f608d9201e152d83` |
| Trace instructions | `ee453a7f6fe43ddabd6d25ddd3bf654781171ec18acf377a11b8c968ec615775` |
| Projection xclbin | `e5c0c3493be210259081e65acfe1d08814e35adb7109713848c8488a61e7edd5` |
| Projection instructions | `76a1fa8f739def18653c8a8bd3d6fdfc2286908b7cd2c37d030fa8b583ccf2d9` |

## Full-model replay

`carry/full` uses the new output projection and normal FFN, the previously
validated compensated LN, and all other baseline kernels. Source fixtures,
reference tensors, seeds and acceptance bounds are unchanged. It completes
3564 NPU dispatches, including three autoregressive tokens and full reset
replay. All resets, state/cache copies, guards and token feedback checks pass.
The tokens remain248045 ->8678 ->198 ->2; minimum logits correlation is
0.9999876023 against0.9999.

Full acceptance remains FAIL:18892/18896 slice checks and4121/4122 decode
checks pass. All five numerical failures are retained:

| Token/layer | Field | maxrel | Required bound |
|---|---|---:|---:|
| cold1/layer47 |head17_og|0.02100840336|<0.02 (cosine also fails)|
| cold1/layer59 |y|0.005450936233|<0.005|
| cold1/layer63 |y|0.02660544621|<0.005|
| cold2/layer47 |head18_og|0.02137096774|<0.02|
| cold1/final |final_norm|0.01282051282|<0.008|

The gate is not globally improved merely because the isolated FFN is more
accurate: the preceding output+FFN+LN replay had three failures. Keep this mode
experimental; do not change the accepted default kernel set or tolerances.

At cold0/layer2 entry norm, the previous19 differences are eliminated; layer2
post norm drops from435 differences to1. Across384 boundaries, the diagnostic
counts820116 differing BF16 values, including12 local norm errors. These counts
overlap with propagated input errors and must not be added. The first changed
boundary is now one propagated difference at cold0/layer1 entry norm; layer0's
two norms and layer1 post norm match exactly.

The remaining first difference is channel2931. Layer0's projout and res1 are
exact there, and all17408 h BF16 values in layer0 match. Its fo is
-0.0063009862788021564 versus-0.0063009848818182945, a difference of
-1.3969838619232178e-9. The residual addition carries precisely that difference
into y. Consequently the next norm rounds to-0.00160980224609375 rather than
-0.0016021728515625. Offline layer0 FFN diagnosis confirms that all its remaining
FFN output error is local to down, not propagated from h.

Next isolate the K17408 down segments and their final FP32 accumulation at
this channel. Do not assume the six fixed layer1 FFN values explain the other
full-model failures or change the norm reference to match the device.
