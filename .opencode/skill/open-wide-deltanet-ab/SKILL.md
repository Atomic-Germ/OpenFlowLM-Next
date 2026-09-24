---
name: open-wide-deltanet-ab
description: Build and validate the separate open WideDeltaNet alpha/beta dispatch on XDNA2 at H=5120 or H=2560 with 48 value heads. Use for banked AB, xn streaming, or diagnosing the fused glue DMA limit; this is not full DeltaNet or model acceptance.
---

# Standalone WideDeltaNet AB on XDNA2

Run from repository root. Results and boundaries are in
`specs/open-engine/plans/wide-deltanet-bringup.md`. This skill records successful
builds and 14 NPU dispatches on Strix, XRT2.26.0, 2026-09-24.

## Build and run

Use `ironvenv` from `ironvenv-requirements.txt` (mlir-aie1.4.2 and pinned Peano).
Python3.14 uses `ironvenv/lib/python3.14/site-packages/llvm-aie/bin`; do not assume
the older Python3.12 path in historical notes. Do not install
`eudsl-python-extras` alongside this wheel: it shadows `aie`.

```bash
source ironvenv/bin/activate
export XILINX_XRT=/opt/xilinx/xrt
export PATH="$XILINX_XRT/bin:$PATH"
export LD_LIBRARY_PATH="$XILINX_XRT/lib:${LD_LIBRARY_PATH:-}"
for width in 5120 2560; do
  WIDE_DN_HIDDEN=$width python open_kernels/build_design.py \
    open_kernels/designs/wide_deltanet/ab.py \
    open_kernels/designs/wide_deltanet/build_ab_h${width}
  python utilities/test-wide-deltanet-ab.py prepare --hidden "$width"
done
```

Build the existing open XRT harness. The installed runtime here has no C++
development headers or unversioned linker symlink. Matching generated headers
exist in the local XRT build (same 2.26.0 package); use them rather than disabling
ABI checks or inventing `version-slim.h`. On another machine set `xrt_headers`
to its matching SDK include directory.

```bash
xrt_headers=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include
mkdir -p open_kernels/harness/out
g++ -std=c++17 -O2 -I"$xrt_headers" -Iopen_kernels/harness \
  open_kernels/harness/run_kernel.cpp -L/opt/xilinx/xrt/lib \
  -Wl,-rpath,/opt/xilinx/xrt/lib -l:libxrt_coreutil.so.2 \
  -o open_kernels/harness/out/run_kernel
for width in 5120 2560; do
  HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel \
    open_kernels/designs/wide_deltanet/build_ab_h${width}/ab.cfg
  python utilities/test-wide-deltanet-ab.py compare --hidden "$width"
done
```

Run on the host with `/dev/accel/accel0` access; this workspace's sandbox hides
the device, so hardware dispatch needs the execution tool's normal escalation.
A Docker substitute needs the device plus a matching XRT/amdxdna runtime.
Do not run two hardware tests concurrently or reset a device used by other work.

`prepare` writes production-packed banked weights, identical quantized inputs
for the float64 reference, and seven dispatches per width. It deletes stale
results and initializes output to NaNs on every run. `compare` checks all heads
for finite results, cosine >0.99999 and max-relative <1e-4 for alpha, beta logits,
decay and beta. Do not relax these inherited dn_glue tolerances. Logs, metadata,
binary fixtures and compiled files belong under the ignored build directories.

## Constraints learned from actual compilation

- Fused glue with separate weights, xn and gact needs **three input DMA
  channels**, but a core has two. Moving shim endpoints does not fix it.
  `utilities/probe-qwen35-wide.py --scope glue` reproduces this failure;
  `--scope layer` also exposes the independent main-core `acquire(3)`/depth2
  failure at H5120 with synthetic FFN8192. This is not a 27B build.
- AB now runs separately with two input/one output channels on Tile(0,2),
  three BOs. Side is bank0 Wa/Wb/small, then bank1 Wa/Wb/small; xn is replayed
  four times. Issue the result drain and weight fill before paced xn fills.
- Weight tiles stay bf16[64,32]. Reuse two fp32[32] accumulators and one
  bf16[2048] xn scratch. A/dt_bias use global heads; accumulators use local
  lanes. Tail bank has only16 active heads. Small constants store A=-exp(A_log).
- Actual placed buffers plus reserved stack occupy21504 B at both widths.
  ELF text is14368/10960 B at H5120/H2560. Inspect
  `final.prj/input_with_addresses.mlir` and the linker script after changes;
  Python buffer arithmetic alone does not establish placement.
- The design specialization hashes source/header dependencies and geometry;
  `build_design.py` clears `final.prj` to prevent stale objects.

## Acceptance boundary

This validates only AB and its nonlinearities. Seven independent AB inputs do
not validate recurrent state. The subsequent chain is documented in
`.opencode/skill/open-wide-deltanet-chain/SKILL.md`, including the A7 procedure
and a known stricter head-local precision limitation. Keep
catalogue promotion, model packing and Flash-specific architecture work behind
the user's A7/full-model gates. No closed kernels or CPU compute fallback.
