# Dense FFN precision gate — 2026-09-24

Follow-up to `f2edf21` and [segmented-dense-ffn.md](segmented-dense-ffn.md).
**The synthetic full FFN now passes the unchanged strict gate on13 inputs**,
with and without up/gate tracing. This closes the previously observed FFN
accuracy failure, not the remaining whole-layer/model acceptance gates.

## First-diverging-tensor investigation

A diagnostic variant of `segmented_probe.py` copies actual ms up/gate to a
separate output FIFO before calling the existing activation. It reuses the
production `ffn_body`/`ffn_sequence` and does not change numerical inputs.
The trace stream carries `[core][band][up64,gate64]`; the host only dumps bytes.
It uses a second output DMA channel, within the physical limit.

The original two-input failure was reproduced with tracing. Up/gate differ
from the independent packed-weight reference by up to5.530479e-6 normalized.
Computing ideal SiLU/product from those device up/gate values localized an
additional activation error up to1.060288e-5. Small h errors then cross bf16
rounding bins before down. Device-h-based down comparisons pass, but never
replace the independent end-to-end output gate.

`vecmath.h`'s multiply splits each fp32 operand into two bf16 components and
keeps three products. Its precision is insufficient at some rounding
boundaries in this wide FFN. The new optional `vecmath_precise.h` retains
three components and six products through the third significance level.
Its exp polynomial and reciprocal Newton iterations use that multiply.
Distinct function names avoid COMDAT collisions with existing helpers.

Only segmented dense `dense_act` selects precise SiLU and its final product.
Q4 GEMV arithmetic, table preparation, packing, segment sums and the base
`vecmath.h` remain unchanged. Generated legacy TUs for FF4096/8192/12288 were
compared byte-for-byte with `f2edf21`; all match. Existing mixed-format/MoE
regressions also pass. No new Q4 decoder or CPU neural fallback was added.

For the original two inputs:

| Check | Before | After |
|---|---:|---:|
| Input0 final max-relative | 4.096821e-5 | 2.285161e-5 |
| Input1 final max-relative | **1.123084e-4 FAIL** | **7.534229e-5 PASS** |
| Worst local SiLU/product error | 1.060288e-5 | 2.141996e-7 |

The original weights and inputs are retained. Pool SHA256 is unchanged:
`6510bc8a84c44d857e25f4245b05d281f68b54ef63147170951e7bd27dfeb08a`.
Additional inputs are generated after the original weights, preserving the
initial RNG sequence instead of replacing the failing cases.

## Comparator correction for tiny vectors

The expanded -1 input produces small nonzero h/output values. The old cosine
expression added1e-30 to the norm product, incorrectly returning approximately
1.95e-16/3.48e-15 for vectors pointing almost identically. Cosine must be
invariant under positive rescaling.

TDD: scale-invariance tests failed at1e-25 and1e-150, then passed after scaling
both vectors by their own maximum before dot/norm. Tests also cover scale1e150,
zeros, nonzero-versus-zero, invalid shapes/NaNs/tails and tiny relative errors.
Max-relative now divides by the actual nonzero reference maximum, without an
absolute epsilon masking errors. Exact zeros are handled explicitly.

Thresholds remain **max-relative <1e-4, cosine >0.9999999** for FFN. The old
full-FFN output still fails when recomputed with this corrected comparator,
confirming that the arithmetic change was needed. Saved WideDeltaNet A7
outputs at H5120/H2560 were recomputed: all416 inherited whole-tensor gates
still pass; the stricter head-local limitations remain. No tolerances were
relaxed or reference tensors substituted.

## Final hardware validation

Real H5120/FF17408 geometry, all eight cores, original packed Q4 weights,
seed38417. Thirteen inputs: original two random inputs, four more random,
zero, +1, -1, impulses at2047/2048/5119, and a repeat of the first input.
CPU comparison uses exact pool bytes and float64 math with the existing bf16
activation interface. Every invocation poisons output regions and checks
exact lengths, finite values and64-byte canaries.

| Tensor | Maximum normalized error | Minimum cosine |
|---|---:|---:|
| up | 4.833310e-6 | 0.9999999999884 |
| gate | 5.530479e-6 | 0.9999999999885 |
| h | 7.068171e-6 | 0.9999999999725 |
| FFN output | 7.534229e-5 | 0.9999999994024 |

Trace: **52/52 gates** (up, gate, h, output). Normal: **26/26 gates**
(h, output). Total final suite:26 NPU dispatches,78 acceptance checks.
All13 h/output tensors are bit-identical with and without tracing; repeated
input is bit-deterministic after intervening inputs. These are synthetic FFN
invocations, not autoregressive model tokens or real layer validation.

## TDD, resources and artifacts

The starting hardware gate was red; the same inputs pass after the math fix.
The expanded suite exposed the independent cosine bug, reproduced by failing
scale tests before its correction. Full regression:

```text
ironvenv/bin/python -m pytest specs/open-engine/tests -q
723 passed, 47 skipped
```

Additional focused recipe/mixed-format checks after the final guard wording
and generator formatting changes:41 passed. The recipe still requires
`OPEN_KERNELS_UNVALIDATED=1` for segmented whole-layer integration.

| Program | Actual buffers + stack/core | ELF text/core |
|---|---:|---:|
| Full FFN | 59392 B | 11712 B |
| Full FFN with up/gate trace | 59904 B | 11872 B |

Placement is from compiler address maps; text from llvm-size. Same Strix
hardware, XRT2.26.0, mlir-aie1.4.2 and Peano21.0.0.2026080301+c9c5ecb7 as
[the preceding report](segmented-dense-ffn.md). Execution used the open host
harness. A baseline trace kernel and two precise FFN variants were built;
no libraries or model packages. No catalogue promotion.

Reproduction is in the new
[precision skill](../../../.opencode/skill/open-dense-ffn-precision/SKILL.md).
Ignored artifacts retain builds, toolchain/spec metadata, hashed fixtures,
traces, intermediate h, run/compare logs and JSON results:

```text
open_kernels/designs/layer_x/build_segmented_precision/ffn_trace  # old math, reproduced failure
open_kernels/designs/layer_x/build_segmented_precise/ffn_trace   # precise,13 inputs
open_kernels/designs/layer_x/build_segmented_precise/ffn         # precise,13 inputs
```

Final xclbin SHA256:

- FFN: `7f33a0a2f66c31e32bcf0effc25f01ef233e15dbad8b606ad7e757d617d193a3`
- trace: `3db820efc3a5db27c7e7fa7952ecac2462738a20e4801bdd9071d5d5786ebf16`

2026-09-25 follow-up: [standalone LN5120](wide-ln.md) now passes hardware
comparison after fixing its input-buffer allocation. This does not fix the
norm embedded in the full layer.

Next: integrate the separate AB/conv/recurrence chain, validate remaining
LN/LM-head/attention points and measure full-layer program placement before
real layer,8-layer slice and64-layer model comparisons. The known fused-glue
DMA blocker and A7 head-local precision caveat remain. Mixed Q8, model
packaging/runtime integration and Flash-specific architecture remain pending.
