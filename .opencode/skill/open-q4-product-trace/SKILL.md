---
name: open-q4-product-trace
description: Capture nine real Q4 product terms and local high/low accumulation to reproduce native FP32 cancellation errors before implementing exact addition.
---

# Nine-term Q4 product trace

Read [the measured report](../../../specs/open-engine/plans/qwen38-q4-product-trace.md).
Prerequisite: the validated real down segment trace from `open-down-segment-trace`.
Use ironvenv Python3.14, mlir-aie1.4.2, llvm-aie21 and XRT. Builds must be
sequential. Preserve original weights, references, tolerances and binary outputs.

```bash
base=open_kernels/designs/wide_deltanet/build_down_products/block53
ironvenv/bin/python utilities/diagnose-wide-down-blocks.py build-products \
  --out "$base" --product-block 53
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-down-blocks.py prepare \
  --source open_kernels/designs/wide_deltanet/build_down_trace/retained/layer0 \
  --kernel "$base" --out "$base/channel4872" --segment 3 --channel 4872
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/channel4872/blocks.cfg"
ironvenv/bin/python utilities/diagnose-wide-down-blocks.py compare --out "$base/channel4872"
```

Use a fresh build directory. `build-products` records the block selection and
binary hashes; avoid calling raw build_design with an undocumented environment
override, which would leave the replay unable to discover its record size.
The same block53 kernel serves channel1769/segment3. Build another fresh kernel
with `--product-block 107` for channel392/segment1.

Each2560-float record contains the previous1120-float block/sum trace followed
by nine160-float product records: operand a, operand b, product, high, low.
These five32-lane planes retain even/odd order. Only the selected tile/block
writes product records; all other product records are zero. The enlarged output
FIFO is single-buffered to preserve L1 headroom. The ninth high/low is checked
against the preceding block trace; zero-state checks must ignore the operands
because weight factors remain nonzero for zero activations.

Observed on all three channels: exact individual products and exact operand
sum; native local compensated accumulation first fails on term7 (`m*xsum_high`).
The expected high is exactly representable in FP32, but the captured high differs
by2^-32 or2^-33 and low compensation remains unchanged. Do not spend further
effort changing block/segment reducers or activation sums based on these cases.

The compact actual operands live in
`specs/open-engine/tests/fixtures/q4_product_cancellation.json`. Use these for
the next exact-add/error-recovery implementation. Do not assume a native
subtraction is exact solely from representability or magnitude ordering.
Both trace kernels occupy6896 B .text; this does not guarantee a correction
fits in the full FFN. Any arithmetic change still needs default regression,
real FFN replay and the unchanged full64-layer acceptance run.

Recorded regression:884 CPU tests pass,47 skip; rebuilt existing corrected
down passes66/66 checks and matches all11 prior activation arenas, segment
traces and instruction bytes with product/block hooks disabled.
