---
name: open-qwen38-recurrence-replay
description: Build private production-layout DeltaNet control and triple-BF16 recurrence kernels, verify byte-exact causal replay, and localize remaining head-output errors without model residency.
---

# Isolated production-layout DeltaNet recurrence

Use to investigate strict head precision after FP32 linear entry/history integration.
This is conditioned on retained production records, not full-model propagation.
No model weights are loaded. Keep production kernels unchanged until the candidate
passes every declared gate and the full-layer resource budget is checked.

## Layout and arithmetic

`utilities/deltanet-replay/` builds a single-head core with the production
10-row slices, 13 slices/head, 130 padded state rows, 5120-byte input elements,
256-byte output elements and 1280-float scratch. State and output have separate
BOs. The input state is read twice; updated output is copied to the next token's
input only after the dispatch. Every head starts at zero; tokens retain their
own candidate state. Record input occupies the first 512 floats of a zero-padded
input element. Output buffers are NaN-poisoned before every dispatch.

The control copies `dnx.h` and generator-owned entry TUs, using the production
`-Os -DDNX_ROWS=10` flags. It must match both production state and output bytes,
including signed zero, at every head/position. A control identity PASS is distinct
from numerical head acceptance: the retained production has 92 failing outputs.

The candidate uses three BF16 components and all nine products in each MAC,
including pass 1, delta, state update, readout and final scale. Delta remains FP32
in private scratch. It uses no native FP32 vector multiply or scalar soft-float
replacement. Nine products do not imply exact FP32 multiplication/accumulation.

## Build and reproduce

Native environment: Python 3.14, mlir-aie 1.4.2,
Peano 21.0.0.2026080301+c9c5ecb7, XRT `/opt/xilinx/xrt`;
AMD Ryzen AI 9 365 / Strix XDNA2. Native AIE compilation and NPU dispatch need
execution-tool escalation on this host. Hardware programs run sequentially.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
for mode in control triple; do
  ironvenv/bin/python utilities/deltanet-replay/build_recurrence.py \
    --kernel-dir Models/qwen38-27b/production-fp32-linear/kernels \
    --out Models/qwen38-27b/deltanet-replay/$mode --mode "$mode"
  ironvenv/bin/python utilities/attention-entry-replay/resources.py \
    --build Models/qwen38-27b/deltanet-replay/$mode \
    --out Models/qwen38-27b/deltanet-replay/$mode-resources.json
done
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/deltanet-replay/recurrence_replay.py prepare \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-linear/kernels \
  --captures Models/qwen38-27b/production-fp32-linear/decode \
  --baseline Models/qwen38-27b/production-fp32-linear/deltanet_state.json \
  --out Models/qwen38-27b/deltanet-replay/inputs
```

Prepare verifies all primary baseline rows, export identity and 289 capture
hashes before extracting 6912 records. Builds pin copied/generated sources and
binary hashes. Configure invalidates old run metadata and deletes expected stale
dumps; do not reconfigure a completed run whose evidence you intend to preserve.

Run these three commands first for `control`, verify identity, then for `triple`:

```bash
mode=control
ironvenv/bin/python utilities/deltanet-replay/recurrence_replay.py configure \
  --inputs Models/qwen38-27b/deltanet-replay/inputs \
  --build Models/qwen38-27b/deltanet-replay/$mode \
  --out Models/qwen38-27b/deltanet-replay/run-$mode
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-fp32-harness/run_kernel Models/qwen38-27b/deltanet-replay/run-$mode/run.cfg
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/deltanet-replay/recurrence_replay.py compare \
  --inputs Models/qwen38-27b/deltanet-replay/inputs \
  --build Models/qwen38-27b/deltanet-replay/$mode \
  --out Models/qwen38-27b/deltanet-replay/run-$mode
```

Preserve complete JSON and exit status. Reference state starts at zero and carries
in FP64, never from captured or candidate state. Each head keeps the unchanged
maxrel <1e-4 and cosine >0.9999999 gates. Padding must remain zero. The control's
`passed`/exit status measures byte identity; `head_precision_passed` and summary
still report its numeric failures. The candidate returns 1 if any head fails.
Readout diagnostics separately compare observed output to exact captured-state
readout and that readout to the independent primary output.

## Measured result — 2026-10-16

Both builds fit one core with stack 6144 and L1 buffer end 22016 bytes. Control
.text 5376; candidate .text 4768. This does not establish full-lx code-size fit.

Control: 6912 dispatches, all 13824 state/output dumps byte-identical to production;
zero padding/state failures, the same 92 head-output numeric failures.
Candidate: 6912 dispatches, all states/padding pass, **4 head outputs fail**:
L25/T0/H10 and L41/T0/H39, H40, H41. Worst state maxrel 8.12033576e-7;
worst output maxrel 4.24254975e-4 (bound 1e-4), failed-output maxabs
1.95648570e-10. This candidate is not ready for production.

All four failures have readout error; L41/T0/H40 also fails when its captured
FP32 state is read out exactly (state-effect maxrel 2.08397795e-4). Next test
improved accumulation and a factored readout that avoids projecting the rounded
updated state; do not assume a readout accumulator fix alone is sufficient.

Evidence: `Models/qwen38-27b/deltanet-replay/`, separate inputs, build/source hashes,
resource audits, two run directories and comparison JSON/status. Production build
key remains `dc6c7d83b46ba6f25fe2cd0a641527737ada1f1de6422bbe156c0d51617b096f`.
Resident runtime/reset, full-depth candidate propagation and model acceptance
remain separate. This experiment does not need the whole model in RAM.

TDD coverage: 19 new tests; full host suite 1265 passed, 47 skipped.
