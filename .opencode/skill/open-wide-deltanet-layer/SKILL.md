---
name: open-wide-deltanet-layer
description: Build the baseline synthetic H5120/FF17408 DeltaNet layer and its prerequisite kernels using byte-only composition. Use for padded state adapters, baseline reproduction and the original precision investigation.
---

# Wide DeltaNet layer acceptance

Read `specs/open-engine/plans/wide-deltanet-layer.md` for the baseline failure.
The passing composition and current commands are in
`.opencode/skill/open-wide-deltanet-precision/SKILL.md`. The commands below
reproduce the baseline (warm-1 maxrel0.00908004), not the precision fix.
Keep that fixture; one-layer acceptance does not imply model support.

## Build

Use the installed ironvenv and XRT as in `open-wide-deltanet-ab/SKILL.md`.
From repository root:

```bash
export XILINX_XRT=/opt/xilinx/xrt
export PATH=/opt/xilinx/xrt/bin:$PATH
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}
base=open_kernels/designs/wide_deltanet/build_layer
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 5120 --projection-n 16384 --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 6144 --projection-n 5120 --out "$base"
LN_N=5120 LN_EPS=1e-6 ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/ln/ln.py "$base/ln_precise"
DN_POST_HEADS=48 ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/dn_post/post.py "$base/post"
DN_POST_HEADS=32 ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/dn_post/post.py "$base/post32"
```

Build the reused primitives if absent, using the exact existing skills:

- `open-wide-deltanet-chain/SKILL.md`: `build_ab_h5120`, `build_glue`,
  `build_step` under `open_kernels/designs/wide_deltanet`.
- `open-dense-ffn-precision/SKILL.md`: the no-trace three-BO FFN under
  `open_kernels/designs/layer_x/build_segmented_precise/ffn`.

Never build two `layer_x` specializations concurrently: generators overwrite
shared `.cc` files. Full FFN uses the original production pool offsets and
8192+8192+1024 down segmentation. Mixed Q8 is not implemented here.

## Run and compare

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-deltanet-layer.py prepare
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$base/acceptance/layer.cfg"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-deltanet-layer.py compare
# Expected exit 1 until the documented full-layer precision blocker is fixed.
ironvenv/bin/python utilities/test-wide-deltanet-layer.py post-prepare
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$base/acceptance/post.cfg"
ironvenv/bin/python utilities/test-wide-deltanet-layer.py post-compare
ironvenv/bin/python utilities/test-wide-ln.py prepare --width 5120 --eps 1e-6 --build-dir "$base/ln_precise"
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$base/ln_precise/ln.cfg"
ironvenv/bin/python utilities/test-wide-ln.py compare --build-dir "$base/ln_precise"
ironvenv/bin/python -m pytest specs/open-engine/tests -q
```

The harness requires `/dev/accel/accel0`; use normal tool escalation if hidden.
Run NPU programs sequentially. `prepare` regenerates deterministic weights,
references and cfg, removes stale outputs, and hashes every fixture and
kernel. Re-prepare after rebuilding. Generated binaries and traces are ignored;
keep logs with the build. The default is two four-token sequences plus reset
repeat:90 dispatches. This is synthetic layer bring-up, not a runtime benchmark.

## Contracts and traps

The layer uses only `run` and byte `copy` between dispatches. Reference arrays
never enter its cfg. Standalone recurrent state has128 rows/head; production
state has140. Copy active rows explicitly and preserve the twelve padding rows.
AB has separate bank-major Wa/Wb regions in production; the standalone BO
interleaves each bank's Wa, Wb and a duplicated4096-byte small-constant block.

Use inherited `layer_x/compare.py` full-layer gates: xn8e-3; residual, xm,
conv/state2e-2; final output5e-3; cosine>0.9999. Conditional GEMV and FFN
comparisons retain maxrel<1e-4, cosine>0.9999999. Conditional checks use device
inputs *after* execution to locate error; they do not replace the independent
whole-layer reference. Results retain intermediate and head-local diagnostics.

Precise streamed LN removes the initial xn discrepancies on this fixture.
Standalone post opts into `POST_PRECISE=1` at both32 and48 heads; the fused
caller's default remains0, preserving its old arithmetic. Legacy fused shim
DMA limitations remain. Neither change resolves the complete layer's warm-1
failure. Do not change the seed, relax the gate, or present cosine alone as a
pass. Investigate remaining projection/conv/state rounding and FFN amplification
on the saved intermediate traces before advancing the validation ladder.
