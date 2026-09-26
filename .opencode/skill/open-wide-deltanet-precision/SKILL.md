---
name: open-wide-deltanet-precision
description: Build and validate the corrected Q4, precise conv and compensated recurrence composition that closes the synthetic H5120 DeltaNet layer gate. Use for full-layer precision regressions and offline boundary diagnosis.
---

# Wide DeltaNet layer precision

Read `specs/open-engine/plans/wide-deltanet-precision.md` for measured results,
failed intermediate experiments and scope. One synthetic DeltaNet layer passes;
the next B7 task is the full attention layer. Runtime, model support and packing
are still pending. The fused wide `lx` DMA guard still applies.

Use `.opencode/skill/open-wide-deltanet-layer/SKILL.md` to build the prerequisite
LN, post, AB, ordinary output projection and segmented FFN artifacts if absent.
Do not rebuild them unnecessarily. The passing composition replaces only QKV/Z,
glue and recurrence. Run from repository root with the installed `ironvenv`.

```bash
export PATH="/opt/xilinx/xrt/bin:$PATH"
base=open_kernels/designs/wide_deltanet
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 5120 --projection-n 16384 --projection-correction \
  --out "$base/build_layer_compensated"
DNGLUE_PRECISE=1 ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/wide_deltanet/glue.py "$base/build_layer_compensated/glue_precise"
DN_HEADS=48 DN_STEP_PRECISE=1 ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/deltanet/dn_step.py "$base/build_layer_compensated/step_precise"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-deltanet-layer.py prepare \
  --qkv-build "$base/build_layer_compensated/projection_k5120" \
  --glue-build "$base/build_layer_compensated/glue_precise" \
  --step-build "$base/build_layer_compensated/step_precise" \
  --out "$base/build_layer_precise/acceptance"
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$base/build_layer_precise/acceptance/layer.cfg"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-deltanet-layer.py compare \
  --out "$base/build_layer_precise/acceptance"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-deltanet-layer.py \
  --out "$base/build_layer_precise/acceptance" --tag warm-1
```

The XRT bin path supplies `aiebu-asm` to direct builds. NPU dispatch needs
`/dev/accel/accel0`; use tool escalation when hidden. Run hardware commands
sequentially. `layer_x` builds also run sequentially because generation shares
source files. Fixture creation and compare take a few minutes on the host.

Expected:874 checks pass, worst final maxrel1.71226e-4 with the unchanged5e-3
limit. The diagnostic utility verifies input/kernel hashes and guard bytes,
then runs offline CPU replays. It never edits hardware output or acceptance
references; its results cannot replace end-to-end acceptance.

All precision switches default off. Q4 correction is currently validated only
for the standalone H5120/K5120 dense Q4 projection. Table22592 bytes and actual
main-core placement63552 bytes leave1984 bytes of L1. The table includes a
512-byte compensation tail for up to four32-row partitions. Prep and GEMV must
use the same macro. Do not enable it globally in other kernels without checking
all table allocations, offsets and DMA layouts. Precise recurrence reuses its
three scratch arrays as FP32; do not mix precise and legacy passes in one build.
Build hashes include the precision switches and relevant included headers.

Regression commands (after the corresponding builds):

```bash
ironvenv/bin/python utilities/test-dense-projection.py prepare \
  --build-dir "$base/build_layer_compensated/projection_k5120"
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel \
  "$base/build_layer_compensated/projection_k5120/projection.cfg"
ironvenv/bin/python utilities/test-dense-projection.py compare \
  --build-dir "$base/build_layer_compensated/projection_k5120"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 5120 --out "$base/build_default_regression"
ironvenv/bin/python utilities/test-dense-projection.py prepare \
  --build-dir "$base/build_default_regression/projection_k5120"
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel \
  "$base/build_default_regression/projection_k5120/projection.cfg"
ironvenv/bin/python utilities/test-dense-projection.py compare \
  --build-dir "$base/build_default_regression/projection_k5120"
DN_HEADS=32 DN_STEP_PRECISE=0 ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/deltanet/dn_step.py "$base/build_default_regression/step32"
ironvenv/bin/python utilities/test-deltanet-step.py prepare --build-dir "$base/build_default_regression/step32"
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$base/build_default_regression/step32/step.cfg"
ironvenv/bin/python utilities/test-deltanet-step.py compare --build-dir "$base/build_default_regression/step32"
ironvenv/bin/python -m pytest specs/open-engine/tests -q
```

Keep baseline and failed experiment directories separate from a new run;
`prepare` clears stale outputs and rewrites its fixture directory. Re-prepare
after rebuilding a referenced kernel. Do not adjust seeds or tolerances to
resolve a regression. Inspect conditional errors, BF16 rounding boundaries and
state propagation before changing arithmetic.
