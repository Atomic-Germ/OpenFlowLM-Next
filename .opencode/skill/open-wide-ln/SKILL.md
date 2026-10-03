---
name: open-wide-ln
description: Build and validate standalone residual RMSNorm at width 5120 with streamed inputs, and regress the legacy 2048/4096 paths. Use for wide LN memory placement, DMA ordering, or repeated-call statistics bugs.
---

# Streamed residual RMSNorm

The H5120 standalone primitive is hardware validated at epsilon 1e-6. This
does not validate the norm worker embedded in `layer_x`, a full layer, or a
model. Keep the catalogue/model gates until their separate acceptance.

From the repository root, with the installed ironvenv/Peano and XRT:

```bash
export XILINX_XRT=/opt/xilinx/xrt
export PATH=/opt/xilinx/xrt/bin:$PATH
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}
ln_build=open_kernels/designs/ln/build_wide_stream
LN_N=5120 LN_EPS=1e-6 ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/ln/ln.py "$ln_build"
ironvenv/bin/python utilities/test-wide-ln.py prepare --width 5120 --eps 1e-6 --build-dir "$ln_build"
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$ln_build/ln.cfg"
ironvenv/bin/python utilities/test-wide-ln.py compare --build-dir "$ln_build"
ironvenv/bin/python -m pytest specs/open-engine/tests/test_wide_ln.py -q
```

The harness resolves cfg-relative paths. Run NPU calls sequentially; device
access may need tool escalation when `/dev/accel/accel0` is sandbox-hidden.
Use the existing open harness; build instructions are in
`../open-wide-deltanet-ab/SKILL.md`. No closed library is needed.

Regression: repeat the commands with width 2048 and 4096 and distinct build
directories, changing **both** `LN_N` and `--width`. Change `LN_EPS` and
`--eps` together. Results, hashed fixtures and canary-protected output dumps
live in the ignored build directory. Preparation clears stale outputs.

The input FIFO has depth 2, but every acquire/release is **one** element.
Sequence: x0, add0, x1, add1, w. Store x temporarily, then replace it by
x+add and accumulate 32 lane sums. Reset sums on half0 every invocation.
Normalize both halves with one global inverse RMS. Emit y0, y1, xn using a
depth-1 output FIFO; post output drains before starting fills.

Do not restore the five-input acquire: the compiler's consumer buffers at
N5120 require 77824 bytes with stack. Current actual placement is 57472 bytes
including stack and scratch; `.text` is 3200 bytes. The helper enforces a
60 KiB budget, so N6144 is explicitly unsupported by this schedule.

Use the existing LN gate: residual relative error <1e-6; bf16 xn relative
error <8e-3, cosine >0.999999, bit mismatches <N//20. Zero vectors need
explicit handling. These bf16 thresholds differ from fp32 FFN gates.
The existing vecmath add is approximate, not bit-exact IEEE addition;
random residual errors remain below 9e-8 at N5120.

Keep `ln_stream.py`, `ln.h` and C++ sources in source/cache dependencies.
Report actual `final.prj/input_with_addresses.mlir` and ELF sections, not
just the helper's estimate. Full evidence and next blockers:
`specs/open-engine/plans/wide-ln.md`.
