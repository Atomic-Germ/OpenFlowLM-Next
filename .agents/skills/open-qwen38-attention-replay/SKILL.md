---
name: open-qwen38-attention-replay
description: Reproduce Qwen3.8 gated attention outputs on the NPU with captured Q/KV/softmax boundaries and compare BF16 versus FP32 output without loading model weights.
---

# Isolated attention precision replay

Use after the independent per-head checker finds absolute-error failures.
`utilities/attention-replay/` reuses the production `xlayer._attn` worker and
attention functions: six cores, four heads each, VEXP=1 and RB=1 for Qwen3.8.
The standalone older `designs/attn/attn.py` has different geometry and is not
this model's control. The replay currently requires split qwen35 attention
with NHL=KVH, RB=1 and VEXP=1.

The control calls the existing BF16 finish and promotes its output to FP32.
The experiment retains the same arithmetic immediately before that final
BF16 store. Both modes expose Q's scaled BF16 hi/lo pair, BF16 K/V rows,
softmax O accumulators and padded [max,sum] state. They run all fixture
positions within each dispatch, carrying hardware KV history. Each layer
starts from zero KV. No projection weight pools or resident full model are
needed; the container is mapped to read only the small norm tensors.

## Build

Use the native environment from the production-preflight skill: mlir-aie 1.4.2,
Peano 21.0.0.2026080301+c9c5ecb7 under Python 3.14, XRT `/opt/xilinx/xrt`.
Build forkserver and NPU execution require sandbox escalation on this host.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
for mode in baseline fp32; do
  ironvenv/bin/python utilities/attention-replay/build.py \
    --spec Models/qwen38-27b/production-fp32-front-kernels/spec.json \
    --mode "$mode" --tokens 3 \
    --out "Models/qwen38-27b/attention-replay/$mode-build"
  ironvenv/bin/python utilities/core_sizes.py \
    "Models/qwen38-27b/attention-replay/$mode-build"
done
```

`build.json` is written only after successful compilation and pins the mode,
token count, source key, geometry and binary hashes. Source identity includes
the production build key, probe code, host layout and build wrapper. The scorer
is excluded from the binary key so diagnostics can be improved without falsely
changing compiled provenance. Do not run source-generating tests during builds.
Every retained core ELF must remain below 16384 bytes.

**DMA lesson:** core 0 cannot have separate KV, output and debug FIFOs: only
two output DMA channels are available. It emits KV followed by debug on the
same BF16-typed FIFO; the host drains those regions into different buffers.
Other cores have output and debug FIFOs. Debug stores raw bytes, not BF16
conversions. Each ExternalFunction has its own C++ source file.

## Prepare, run, score

The harness from `/tmp/qwen38-fp32-harness` may need recreating; use the
production-preflight skill's native build instructions. For each mode:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/attention-replay/check.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-front-kernels \
  --fixture Models/qwen38-27b/production-fp32-front-multitoken \
  --build Models/qwen38-27b/attention-replay/baseline-build \
  --out Models/qwen38-27b/attention-replay/baseline --mode baseline
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-fp32-harness/run_kernel Models/qwen38-27b/attention-replay/baseline/run.cfg
```

Repeat the preparation command with `--score`. Then repeat the sequence for
`fp32` with matching build/output paths and mode. Both recorded scores return
**exit 1** because the primary numerical gate still fails. Hardware exit 0
only establishes that the dispatches completed. `--layers 59,63` narrows the
selection; by default every covered full-attention layer and token is required.

Preparation verifies the fixture/export and compiled mode/source identities,
compares model Q/K norm weights with the captured constants, and pins all
inputs, constants, position table, configuration and binaries. Scoring requires
exact hash coverage and sizes/finite values, and invalidates stale scores
before validation. It rejects divergent Q copies across cores or invalid
softmax sums. Compare the two modes' input, KV and debug files byte for byte.
The control must reproduce production outputs bytewise.

The primary oracle remains unrounded FP64 QK RMSNorm/RoPE/causal GQA/sigmoid
from retained projections. Every head requires **absolute max error <0.02**
and **cosine >0.9999**. Captured-QKV and captured-KV-only comparisons are
conditional diagnostics and cannot override primary failure. The KV-only
reference recomputes ideal Q independently, so it isolates KV representation
from Q normalization and subsequent attention arithmetic.

## Recorded result — 2026-10-12

Both modes ran 16 layer dispatches, each containing three positions, covering
1152 heads. The control reproduces all production outputs bytewise, including
43 failing heads and worst absolute error 0.2226126094. FP32 output reduces
failures to **21**, in layers 51/59/63, worst **0.1974321142** at L59/T2/H9.
All cosine gates pass; FP32 minimum 0.9999418362. Primary acceptance remains
**FAIL**. Inputs, K/V and all debug bytes are identical across the two builds.

Conditional FP64 diagnostics with captured hardware Q/K/V:

- Softmax/PV: all heads pass, worst absolute error **0.0002344581** in both modes.
- Final output: control has 23 failures, worst **0.1096219261**;
  FP32 has none, worst **0.0003921240**.
- Captured KV but independently recomputed ideal Q: FP32 passes every head,
  worst **0.0003919703**. Thus the remaining failures are localized to KV
  representation at these observed boundaries; changing only the output dtype
  is insufficient. This does not validate a corrected KV path.

Core .text maxima: control **15968**, FP32 **15680** bytes. These include
instrumentation and do not prove a complete production layer fits a future
precision change. Production kernels, layouts, model and packing are unchanged.
Evidence is under ignored `Models/qwen38-27b/attention-replay/`: build/hardware
logs, mode-specific build/audit/score JSON and captures, `results.json` and
`build-audit.json`. Next isolate a higher-precision KV path before integration;
full-model regression, recurrent state, prefill and serving remain separate.
