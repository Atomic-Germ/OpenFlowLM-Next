---
name: open-wide-lm-head
description: Build and validate the standalone Q8 LM head at K5120 with the full 248320-token vocabulary, production pool packing, bounded FP64 reference decoding and a K4096 regression.
---

# Wide Q8 LM head

The existing kernel handles K5120 without arithmetic or scheduling changes.
The full-vocabulary synthetic hardware gate passes; complete-model support
and mixed-Q8 layer projections are separate gates. This Q8 head is already
the qwen35 recipe's normal head format.

From the repository root, using ironvenv, installed Peano, and the existing
open harness (build details in `../open-wide-deltanet-ab/SKILL.md`):

```bash
export XILINX_XRT=/opt/xilinx/xrt
export PATH=/opt/xilinx/xrt/bin:$PATH
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}
lm_build=open_kernels/designs/lm_head_q8/build_wide_5120
LMHEAD_K=5120 LMHEAD_N=248320 LMHEAD_CORES=8 ironvenv/bin/python \
  open_kernels/build_design.py open_kernels/designs/lm_head_q8/lm_head_q8.py "$lm_build"
ironvenv/bin/python utilities/test-wide-lm-head.py prepare \
  --width 5120 --rows 248320 --cores 8 --build-dir "$lm_build"
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$lm_build/lm.cfg"
ironvenv/bin/python utilities/test-wide-lm-head.py compare --build-dir "$lm_build"
ironvenv/bin/python -m pytest specs/open-engine/tests/test_wide_lm_head.py -q
```

Match all build geometry variables to fixture arguments. The weight fixture
is 1350860800 bytes; it is generated in complete-band batches through the
production `pack.apply_op(lmhead_q8)`. Reference dequantization uses a memmap
with bounded temporary batches. Preparation clears stale outputs, poisons
the result and adds a canary; comparison verifies hashes including weights.
The harness resolves paths relative to its cfg. Run NPU invocations
sequentially; use normal tool escalation if the device is sandbox-hidden.

For regression, use a separate build directory with K4096, N1152, 8 cores
and matching preparation arguments. This gives nine bands over eight cores,
including a two-band core and seven one-band cores. Full K5120 uses
243 bands on four cores and242 on four cores. Check both whole output and
per-core metrics: normalized max error <1e-4, cosine >0.9999999. Zero has
explicit handling. Repeat-first output must be bit-identical.

The oracle in `lm_head_q8/make_test.py` now derives K from x, rather than
assuming32 chunks per band. Independent tests encode file raster bytes and
compare against dense FP64 GEMV at K2048/2560/4096/5120. Do not use a full
dequantized 248320x5120 matrix just to compute the reference.

Actual K5120 placement: 61696 bytes/core including stack, and2432 text bytes.
This fits physical64 KiB with3840 bytes spare but exceeds the layer recipe's
conservative60 KiB budget by256 bytes. It is a separate primitive, with no
layer scratch; do not extrapolate that placement to full-layer kernels.
The broadcast x and8 weight streams use5 shim columns. Do not increase K or
scratch solely because this configuration compiled.

Artifacts, exact hashes, metrics and remaining blockers:
`specs/open-engine/plans/wide-lm-head.md`.
