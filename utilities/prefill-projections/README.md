# Real-weight prefill projection checks

This diagnostic runs the five production Qwen3.8-27B GEMM shapes with actual
weights from layers 0 and 3 and three captured production decode inputs per
projection. It is a conditioned GEMM check, not a causal 256-token prompt or
full-model prefill acceptance. No full-model NPU residency is needed.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-projections/check.py prepare \
  --out Models/qwen38-27b/prefill-projections/final \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-consistent/kernels \
  --fixture Models/qwen38-27b/production-consistent/decode \
  --pools Models/qwen38-27b/production-full/pools \
  --original Models/qwen38-27b/ffn-projections/run/audit.json
OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  ironvenv/bin/python utilities/prefill-projections/check.py run \
  --out Models/qwen38-27b/prefill-projections/final \
  --harness /tmp/qwen38-consistent-harness/run_kernel
```

Build the current `open_kernels/harness` target with CMake before running.
Use the complete, previously audited kernel export; this checker builds no new
kernels. `load_bundle` verifies compatibility and all export binary hashes.
The original audit must contain the model, selected pool and capture hashes;
preparation refuses drift instead of replacing their provenance.

Each layer contributes entry, output, FFN up/gate and FFN down projections.
Weight packing follows the manifest and must match the retained production
pool/consts bytes exactly. Independent FP64 products decode those same packed
Q4 weights, once with original inputs and once with BF16-rounded inputs.
This checks effective packed weights; it does not independently establish
source-model conversion accuracy. Linear OG captures use BF16, full-attention
OG and both norm/FFN boundaries use FP32. The linear OG gate therefore starts
with an already BF16-rounded input.

The three inputs occupy slots `[0,127,255]` and then `[31,128,254]`; all other
columns are zero. All eight blocks are dispatched twice (16 dispatches), each
in its own process with only one weight/input/output buffer resident.
Repositioning must preserve output bytes. Each constituent projection (including
separate linear Q/K/V/Z, attention Q/K/V/gate and up/gate) and each active column must satisfy both
`rel_fro <= 0.005` and `maxrel < 0.005` against **both** references. The existing
GEMM Frobenius gate is preserved, with an additional maximum-error gate.
Matching the BF16-input control cannot replace the original-input comparison.
All inactive columns must be exactly zero. Partial shapes and nonfinite data
are rejected; there is no truncation to a common array length.

`audit.json` pins the source/build identity, input/capture/model hashes, original
audit, configurations and references. `report.json` records per-role/per-column
metrics, output hashes and slot identity. Before each run the old report and
all expected output dumps are invalidated; a failed run cannot reuse an old
success. Preparation and execution logs, inputs, references and output dumps
remain under the ignored model directory.

The Linux run on 2026-10-18 used the existing production consistent-sum export
and a current, up-to-date CMake `run_kernel` target. No production kernel,
weight conversion or runtime arithmetic was changed. At most **0.1395 GiB**
of logical NPU weight/input/output buffers were resident per process; this
excludes XRT context overhead and CPU memory used to construct references.

Final result: **16/16 dispatches**, **96/96 per-role/per-column checks** against
each reference, exact slot identity in all eight projection blocks, and zero
inactive columns. Worst original-input maxrel / rel_fro:
**0.003121364 / 0.002404212**. BF16-input control:
**0.002682617 / 0.001519963**. All pass the unchanged 0.005 thresholds.
Host regression: **1389 passed, 47 skipped**; native CTest **4/4**.
