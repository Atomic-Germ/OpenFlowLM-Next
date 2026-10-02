---
name: open-wide-attention-layer
description: Build and validate the complete synthetic H5120/FF17408 attention layer with production Q4 packing, gated Q24/KV4 attention, segmented FFN and persistent device KV cache. Use for B7 layer regressions and preparation of an eight-layer slice.
---

# Complete wide attention layer

Read `specs/open-engine/plans/wide-attention-layer.md` for evidence and scope.
The synthetic one-layer gate passes; the next task is an8-layer slice. This
does not enable a fused runtime or declare Qwen3.8-27B supported.

Prerequisites are the existing artifacts from `open-wide-attention` (Q24,
257 streamed rows, including `attention-fixture.json`) and
`open-wide-deltanet-precision` (precise LN, K6144 output projection and segmented
FFN). Use those skills to build missing prerequisites. The only new artifact is
the corrected K5120/N14336 Q/K/V/gate projection. From repository root:

```bash
export PATH="/opt/xilinx/xrt/bin:$PATH"
base=open_kernels/designs/attn/build_layer
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 5120 --projection-n 14336 --projection-correction --out "$base"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-attention-layer.py prepare \
  --out "$base/acceptance"
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$base/acceptance/layer.cfg"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-attention-layer.py compare \
  --out "$base/acceptance"
ironvenv/bin/python -m pytest specs/open-engine/tests -q
```

Run `layer_x` builds sequentially: they share generated source files. Run NPU
dispatches sequentially and use normal tool escalation for `/dev/accel/accel0`
if hidden. Host preparation/comparison take a few minutes. The default run uses
seed38428, rows257, four cold and four warm tokens and one reset repeat:
63 dispatches,458 passing checks, maximum final error1.66353e-5 versus5e-3.

`--attn-build` and `--rows` must agree with the attention primitive fixture and
its hashed binaries; preparation validates this before writing data. The
`--projection-build` directory must have the build tool's metadata for
K5120/N14336/FF17408. The preceding DeltaNet N16384 binary is incompatible with
this output BO. Do not refresh primitive hashes manually to bless stale builds.

Packing uses the actual Qwen35 plan from synthetic native Q4 chunks, then
cross-checks each matrix with the independent GGUF pool packer. Preserve the
two halves of fused `q_proj`: the combined result is Q|K|V|gate, while attention
reads separate Q|gate and K|V buffers. Constants come from `CA_*`. The layer
captures `AA_*` activations; the reused FFN probe needs explicit copies between
AA_XM/AA_OUT2 and A_XM/A_OUT2. Never equate these offsets.

The cache is interleaved BF16 `[rows,2,4,256]`; each device-produced4096-byte
row is copied to `pos*4096`. The oracle independently updates its own cache.
Future rows remain NaN, and the entire device cache plus guard is checked after
every token. Reference files must never appear in the harness cfg. Keep the
head-local output/K/V gates, strict conditional GEMV/FFN checks, poisoned BOs,
hashes and reset/repeat checks. Do not tune seeds or tolerances to pass.

Projection data placement is63552 bytes/core including stack, text4368 bytes.
Attention reuses51728 data/14400 text bytes/core. These contexts are sequential;
their resource totals do not demonstrate that a fused layer fits. The runtime
DMA patcher, model container, long-context model performance, full-model and
autoregressive checks remain separate work. No new closed dependency is used.
