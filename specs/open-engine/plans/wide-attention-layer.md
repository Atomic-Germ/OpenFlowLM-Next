# Complete wide attention layer — 2026-09-27

**B7's synthetic one-attention-layer gate passes:458/458 checks.** This follows
the [accepted DeltaNet layer](wide-deltanet-precision.md). The next stage is
the8-layer slice with distinct per-layer weights and persistent state. Model
integration,64-layer execution, autoregressive decoding, runtime export and
catalogue promotion remain pending.

## Composition and TDD

The H5120/FF17408 all-Q4 layer uses seven NPU dispatches per token: entry
RMSNorm, Q/K/V/gate projection, gated attention, output projection, residual
RMSNorm, segmented FFN, final residual addition through the LN primitive.
The attention tuple is Q24/KV4/HD256/ROT64. No host neural math executes between
dispatches; the harness only copies BO bytes and runs the kernels.

`recipes/wide_attention_layer.py` defines the composition and adapters. The
new corrected K5120/N14336 projection produces `[Q | K | V | gate]`; explicit
copies form the attention worker's `[Q | gate]` and `[K | V]` BOs. The fixture
uses the actual Qwen35 pack-plan, including the two source halves of fused
`q_proj`, and independently checks every packed matrix against the GGUF Q4
packer. Constants use the production `CA_*` regions. Captured attention
activations use `AA_*`; adapters bridge `AA_XM/AA_OUT2` to the reused segmented
FFN probe's different `A_XM/A_OUT2` offsets.

Eleven new tests first failed for missing adapters/composition, then passed.
They check every projection range, production pack-plan regions, all seven
dispatches, cache boundary positions and rejection of unsupported geometry.
Two further RED/GREEN tests ensure preparation rejects mismatched static
attention DMA row counts, stale attention binaries, and projection output sizes
that would overrun the result BO. Total13 new tests; full open-engine regression:
**773 passed,47 skipped**.

## Numerical and state contract

`utilities/test-wide-attention-layer.py` uses seed38428, synthetic Q4 weights,
BF16 effective norm weights and an independent FP64 reference. It rounds only
at actual FP32/BF16 boundaries. Q/K head normalization, partial RoPE, GQA,
softmax and sigmoid gating reuse the independent attention oracle. The new
cache row and gated output are BF16 before subsequent reference computation.

Four cold tokens run at positions0–3. Four warm tokens start with253 populated
cache rows and run at253–256. All unused future rows are NaN. Within each
sequence the NPU-produced KV row is copied into the device cache; the oracle
updates its separate cache and never supplies intermediate state to the NPU.
Each token checks the entire cache byte-for-byte against the previous device
cache plus its new row, including untouched past/future rows and tail canary.
The final cold reset repeats output and cache exactly.

The fixture hashes kernels, weights, inputs, references and cfg. Outputs start
as NaN and have64-byte canaries. Comparison rejects changed fixtures/artifacts
or damaged guards. Intermediate/conditional results remain in the result JSON.

Unchanged thresholds: entry xn8e-3; attention residual1e-2 and xm2e-2 from
`layer_x/compare_ax.py`; K/V1e-2 and gated attention output2e-2, also checked
per head; final output5e-3 from the full-layer gate. All require cosine>.9999.
Conditional Q/K/V/gate/out/down and full FFN checks use maxrel1e-4 and
cosine>.9999999 on device inputs after inference. Conditional checks never
replace the independent end-to-end reference. Direct device V conversion is
also checked bit-for-bit against device-projected V.

## Results

Eight token evaluations plus one reset repeat execute63 NPU dispatches.
All458 checks pass, including192 output-head checks,32 K-head checks and32
V-head checks. No numerical thresholds, seeds or references were adjusted.

| Quantity | Worst normalized max error | Minimum cosine |
| --- | ---: | ---: |
| Entry xn | 0 | 1 |
| Attention residual | 2.14338e-5 | 0.999999999943 |
| FFN input xm | 5.78035e-3 | 0.999999958419 |
| Gated attention output | 3.28947e-3 | 0.999999909304 |
| Gated output, per head | 4.54545e-3 | 0.999999297877 |
| New K, per head | 3.57143e-3 | 0.999999885399 |
| New V, per head | 6.76407e-5 | 0.999999999886 |
| Final output | **1.66353e-5** | **0.999999999943** |
| Conditional Q/K/V/gate | 1.89446e-7 | 0.999999999999993 |
| Conditional output projection | 2.48957e-6 | 0.999999999999695 |
| Conditional FFN | 5.75639e-5 | 0.999999999916 |
| Conditional down | 4.48087e-5 | 0.999999999973 |

The synthetic FFN has both large-output and near-zero-output cases; therefore
the strict scale-invariant conditional FFN gate is retained alongside the layer
output gate. The input x for each token is independent synthetic data: these
are persistent-state decode steps, not an autoregressive/model-quality claim.

## Resources and reproduction

The only new binary is the corrected K5120/N14336 projection. Addressed MLIR
places its22592-byte table at6144 and all buffers plus stack end at63552 bytes
per core; ELF text is4368 bytes. It uses eight cores and28 output bands/core.
The production attention worker is reused from `build_wide_257`: six cores,
51728 data bytes/core and14400 text bytes. Existing precise LN, K6144 output
projection and segmented FFN artifacts are reused unchanged.

These are sequential standalone contexts, **not** a measured fused full-layer
placement. The fixed257-row attention stream does not validate the runtime
instruction patcher or long-context model throughput. The existing fused-wide
DMA guard remains. No converter format or catalogue entry changes are needed
for this synthetic gate, and no packed model is produced.

Commands: [open-wide-attention-layer skill](../../../.opencode/skill/open-wide-attention-layer/SKILL.md).
Artifacts under `open_kernels/designs/attn/build_layer/`: `projection_k5120/`
and `acceptance/`. Logs: `/tmp/attention-layer-{projection-build,prepare,hardware,compare,unit}.log`.
Toolchain: Python3.14.4, mlir-aie1.4.2,
llvm-aie21.0.0.2026080301+c9c5ecb7, XRT2.26.0; Ryzen AI9 365/Strix,
firmware1.1.2.64. Validation ran on the host through the open XRT harness.

SHA256 under `acceptance/`:

```text
41e19ec80cad86c6fce9cd7c5baac2234224bd26fc166948c36235eac1992160  layer-fixture.json
9224fcee9e70b0d21b9821435a1e9fd62ae343c2eee4192a4b891726abba05e6  layer-results.json
```
