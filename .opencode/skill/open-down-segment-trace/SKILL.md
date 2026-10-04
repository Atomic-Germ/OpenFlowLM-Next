---
name: open-down-segment-trace
description: Capture real down GEMV segment high/low values and distinguish intra-segment error from segment reduction without linking SiLU.
---

# Down-only high/low trace

Read [the report](../../../specs/open-engine/plans/qwen38-down-segment-trace.md).
Continue from `open-down-rne`. The important result is that channel 4872 is
already wrong inside segment 3, K=12288..16383. Its cross-segment reduction is
exact for the captured operands. A larger segment reducer cannot fix it.

Use ironvenv (Python 3.14, mlir-aie 1.4.2, llvm-aie 21), XRT and the existing
harness. Build sequentially; the generators share translation units. Preserve
old binaries, real weights, immutable references and acceptance thresholds.

```bash
base=open_kernels/designs/wide_deltanet/build_down_trace/retained
OFLM_KEEP_FAILED=1 ironvenv/bin/python utilities/probe-qwen35-wide.py \
  --scope down --ffn 17408 --ffn-correction --product-correction --block-carry \
  --segment-carry --down-rne --down-trace --out "$base"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-segmented-dense.py \
  prepare --build-dir "$base/down"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/down/segmented.cfg"
ironvenv/bin/python utilities/test-segmented-dense.py compare --build-dir "$base/down"
```

The trace stores `[core, segment, band, plane, lane]`, four 64-float planes:
segment high, segment low (unpermuted from the Q4 carry table), accumulated high,
accumulated low. It observes the production accumulator without changing its
inputs. `wide_down_trace.decode_trace` restores `[segment, plane, channel]`.
All work on weights/activations remains on the NPU; FP64 host calculations are
conditional diagnostics only. Scratch geometry remains the full layer's.

Replay the accepted real full-FFN trace, in a NEW output directory:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-down.py prepare \
  --source open_kernels/designs/wide_deltanet/build_residual_precision/final_rne_layer0 \
  --kernel "$base/down" --out "$base/layer0"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/layer0/down.cfg"
ironvenv/bin/python utilities/diagnose-wide-down.py compare --out "$base/layer0" \
  --channel 4872 --channel 392 --channel 1769
```

Preparation requires the synthetic gate, original fixture hashes, exact BF16 h
and deterministic source trace. Comparison checks guards, finite values, hashes,
repeat, preserved h, trace/output identity and final equality with full FFN.
`passed` here means the diagnostic faithfully reproduces the source, NOT that
the down projection is accurate or the model passes. `fp32_differences` and each
channel's errors remain explicit. No full-model replay is needed for a purely
observational trace; a subsequent arithmetic change requires the unchanged full
acceptance gate.

Use `llvm-size -A` on `final.prj/elfs_main_core_*/*.elf`. This down-only trace
fits comfortably; it is not evidence that a correction will fit beside SiLU in
the full FFN. The next probe should trace Q4 block accumulation inside segment 3,
particularly its high/low normalization and product correction at channel 4872.

Recorded results: retained trace 10016 B .text, 66/66 synthetic checks; default
K8192 down regression 33/33 checks. Real replay matches the full FFN, both
repeats and trace/output reconstruction exactly, with three FP32 differences
from the independent down reference. CPU suite: 877 passed, 47 skipped.
