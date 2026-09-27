# Wide eight-layer slice — 2026-09-27

**B7's synthetic8-layer slice passes3098 checks across370 NPU dispatches.**
It follows the [DeltaNet](wide-deltanet-precision.md) and
[attention](wide-attention-layer.md) one-layer gates. The next milestone is
multi-token decode with token selection; this test uses independent synthetic
hidden inputs and is not autoregressive or full-model acceptance.

## Composition and TDD

`recipes/wide_slice.py` reads the first eight entries of the actual ModelSpec
`layer_types`: linear, linear, linear, full, linear, linear, linear, full.
It does not infer the order from an interval. Four tests initially failed for
the missing module, then passed: explicit-order selection, isolated state names,
operand-only command scoping and rejection of unsupported prefixes/indices.
The permutation test moves a full layer to index0 to distinguish array-based
selection from hardcoded every-fourth scheduling.

Each layer owns a distinct Q4 pool, constants and recurrent state or KV cache.
Eight independent RNG streams derive from seed38429 plus the layer index.
Both production pack-plans are exercised from native synthetic Q4 chunks;
each packed matrix is cross-checked against the independent GGUF-to-pool
packer. All eight pool hashes and all eight constant hashes differ.

Weights stream into reusable BOs before each layer; shared kernels reuse their
existing contexts. Scratch BOs are separate for the two layer types. Six
`stateN` BOs and two `cacheN` BOs retain device results between tokens. The
output of each layer is copied directly to the next layer's input. Only the
first layer receives the synthetic host input. Harness commands contain no
reference loads or host neural computation.

The composition reuses the accepted corrected Q4, precise DeltaNet glue/step,
LN/post, gated attention and segmented FFN binaries. Preparation checks the
one-layer result status and kernel hashes, attention DMA geometry and the
projection output shape. **No new xclbin or library was built.**

## Fixture and independent oracle

`utilities/test-wide-slice.py` prepares, hashes and compares the complete
sequence. `utilities/wide_slice_reference.py` evaluates each layer in FP64
with rounding at the same physical FP32/BF16 boundaries as the one-layer
oracles. Its14 shared DeltaNet outputs and10 shared attention outputs were
checked bit-for-bit against the accepted cold-0 one-layer references before
using it for the chained oracle. Each reference layer consumes the preceding
reference output and its own prior reference state, never device feedback.

Default cases are two cold tokens at positions0/1 and two warm tokens at
255/256. Warm recurrent states are independently randomized; each attention
cache starts with255 populated rows and NaNs in future rows. Six DeltaNet
layers use10 dispatches each and two attention layers use7:74 per token.
After all four tokens, all eight states reset and the first complete token
runs again, giving370 dispatches in total.

Every layer boundary records the actual input and output. Comparison checks
input bytes against the previous device output, and incoming state against
that layer's previous device state. It also checks padded DeltaNet state
adapters, zero padding, convolution shifts, activation copies, and the entire
attention cache including untouched rows. Outputs and state have64-byte guards.
The reset repeat compares every captured output/scratch BO and all eight states
bit-for-bit, with guard checks. Kernels, weights, inputs, references and cfg are
hashed; modified evidence is rejected.

## Results and limits

All3098 acceptance checks pass. Full open-engine regression:
**777 passed,47 skipped**. Existing numerical gates are unchanged: xn8e-3;
DeltaNet residual/xm/conv/state2e-2; attention residual1e-2, xm/output2e-2,
K/V1e-2; every layer output5e-3; cosine>.9999. Attention output and new K/V
are also checked per head.

| Quantity | Worst normalized max error | Minimum cosine |
| --- | ---: | ---: |
| Entry xn,32 layer evaluations | 6.32911e-3 | 0.999999920366 |
| Mixer residual | 1.32890e-4 | 0.999999999356 |
| FFN input xm | 6.36943e-3 | 0.999999918119 |
| DeltaNet convolution state | 5.49451e-3 | 0.999999941502 |
| DeltaNet recurrent state | 3.04334e-3 | 0.999999897754 |
| Attention output, per head | 6.94444e-3 | 0.999994204443 |
| Attention new K, per head | 5.91716e-3 | 0.999999881403 |
| Attention new V, per head | 5.74713e-3 | 0.999999874468 |
| Any layer output | 1.35555e-4 | 0.999999999430 |
| **Final eighth-layer output** | **4.11248e-5** | **0.999999999430** |

The stricter diagnostic DeltaNet state head-local1e-4 gate passes only8 of24
layer-token tensors. All24 pass the inherited2e-2 whole-state acceptance gate.
This reflects accumulated deviations through the independent chain and is
retained in `diagnostics`; it is not reported as strict primitive equivalence.
No diagnostic limit, seed or acceptance threshold was changed to pass.

Synthetic Q4 weights can produce nearly inactive branches on some inputs;
the other cases exercise large FFN updates. This fixture tests composition,
state isolation and numerical propagation, not trained-model quality. It has
no embedding lookup, final LM head or token selection, and does not validate
64 layers, model packing, runtime export or long-context performance. Those
remain separate B7/B8 gates. No catalogue entry or converter format is changed.

## Reproduce and resources

Commands and operational constraints:
[open-wide-slice skill](../../../.opencode/skill/open-wide-slice/SKILL.md).
Generated evidence: `open_kernels/designs/wide_deltanet/build_slice/`, containing
eight `layerN/` directories, `slice.cfg`, fixture/results JSON, references and
device traces. Logs: `/tmp/wide-slice-{prepare,hardware,compare,unit}.log` and
`/tmp/wide-slice-oracle-regression.log`.

The harness declares424205768 bytes of BO storage, excluding XRT/kernel
overhead. Packed pool/constant files total1988808704 bytes; they stream through
reused buffers rather than all being resident device BOs. The ten kernel
contexts execute sequentially. This does not demonstrate fused layer placement
or provide a model latency benchmark.

Host/toolchain unchanged: Ryzen AI9 365/Strix, firmware1.1.2.64, XRT2.26.0,
Python3.14.4, mlir-aie1.4.2, llvm-aie21.0.0.2026080301+c9c5ecb7.
Validation uses the open XRT harness; no closed kernel or CPU inference fallback.

SHA256:

```text
8cf65dd07d02d1b65d83a2b7de9877af3461d0176a9b6a92a91958911816fbe2  slice-fixture.json
83325a0b8cfd0e2c493e386861c13805033d8fbdac7e3ca80e8ef04440906bc2  slice-results.json
```
