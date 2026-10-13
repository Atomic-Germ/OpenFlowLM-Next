---
name: open-wide-deltanet-chain
description: Build and test the open 48-head WideDeltaNet AB, conv/record and recurrent-state chain on XDNA2. Use for the A7 synthetic gate at H5120/H2560, persistent-state debugging, or dn_step head-count regressions; not full-model acceptance.
---

# WideDeltaNet recurrent chain

See `specs/open-engine/plans/wide-deltanet-a7.md` for the measured acceptance
boundary and known head-local precision limitation. The preceding AB skill at
`.opencode/skill/open-wide-deltanet-ab/SKILL.md` supplies toolchain and open
XRT harness setup. Use the same `ironvenv` and `run_kernel` binary.

## Reproduce the build and acceptance

From repository root, after activating ironvenv:

```bash
export XILINX_XRT=/opt/xilinx/xrt
export PATH="$XILINX_XRT/bin:$PATH"
export LD_LIBRARY_PATH="$XILINX_XRT/lib:${LD_LIBRARY_PATH:-}"
python open_kernels/build_design.py open_kernels/designs/wide_deltanet/glue.py \
  open_kernels/designs/wide_deltanet/build_glue
DN_HEADS=48 python open_kernels/build_design.py open_kernels/designs/deltanet/dn_step.py \
  open_kernels/designs/wide_deltanet/build_step
for width in 5120 2560; do
  WIDE_DN_HIDDEN=$width python open_kernels/build_design.py \
    open_kernels/designs/wide_deltanet/ab.py \
    open_kernels/designs/wide_deltanet/build_ab_h${width}
  python utilities/test-wide-deltanet-chain.py prepare --hidden "$width"
  HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel \
    open_kernels/designs/wide_deltanet/build_chain_h${width}/chain.cfg
  python utilities/test-wide-deltanet-chain.py compare --hidden "$width"
done
```

Hardware dispatch needs access to `/dev/accel/accel0`; use the execution tool's
normal escalation if the sandbox hides it. Run the hardware programs sequentially.
Keep logs under the ignored build directories. `prepare` clears stale outputs
and records kernel hashes; re-prepare after rebuilding any kernel. All three
dispatches use the open XRT harness, not a model engine or closed capture.

Each width runs cold and warm sequences of8 tokens. The NPU produces AB,
48 records, shifted conv state, recurrent state and output. Host operations
between dispatches are byte copies only. Reference files never enter the run
program. All outputs are NaN-poisoned and have64-byte canaries on each token.
The comparison checks all states/records at every token, not just final output.

## Numerical contract

The original dn_glue/deltanet gates use max-absolute error divided by the
**whole tensor/field's** max-absolute reference. Keep max-relative <1e-4;
cosine >0.99999 for glue and >0.9999999 for recurrence; conv state is bit-exact.
Do not truncate arrays to a common length or ignore incomplete tails.

The optional head-local diagnostic is stricter and is **not satisfied** for
some near-zero outputs on the first cold token: up to1.029e-3 relative at
H5120 and6.639e-4 at H2560. Both inherited split-bf16 glue and recurrence math
contribute when q·k is nearly zero. `chain-results.json` preserves every
head's metrics and reports `head_diagnostics_passed: false`, while the original
whole-tensor gates pass. Do not advertise per-head relative accuracy <1e-4,
discard this diagnostic, change fixtures to avoid it, or loosen tolerances.
Keep it visible when validating real layers/models.

## Resource and layout traps

- Conv side layout is `[AB f32[4,48], pad to4096][conv bf16[10,4,1024]]`.
  Copy exactly768 bytes of AB output to side offset0; no nonlinear recomputation.
- Glue uses two input channels. Lowering allocates3 side,6 activation and4
  output FIFO buffers. Actual buffers plus stack occupy59776 B. The attempted
  fused weights+xn+gact topology still cannot place (three input DMA channels).
- Forty-eight heads share16 key heads: use the derived group size. Record
  boundaries31/32 and the last head47 must be checked. Existing `glue_emit`
  already derives this mapping from compile-time geometry.
- `DN_HEADS=48` builds six heads per core across eight cores; the default is32.
  The per-head C++ math remains unchanged. Actual buffers+stack43776 B/core.
  Two streamed passes read the old state; write the next state into a separate
  BO, then copy it into the next input. In-place S input/output aliasing is not
  validated. The standalone state has128 rows, unlike dnx's padded state layout.
- Hidden width affects only AB. Glue and recurrence xclbins are shared by both
  H5120 and H2560. Catalogue/model support does not follow from this equality.

## Default32-head regression

```bash
DN_HEADS=32 python open_kernels/build_design.py open_kernels/designs/deltanet/dn_step.py \
  open_kernels/designs/wide_deltanet/build_step_legacy32
python utilities/test-deltanet-step.py prepare \
  --build-dir open_kernels/designs/wide_deltanet/build_step_legacy32
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel \
  open_kernels/designs/wide_deltanet/build_step_legacy32/step.cfg
python utilities/test-deltanet-step.py compare \
  --build-dir open_kernels/designs/wide_deltanet/build_step_legacy32
python -m pytest specs/open-engine/tests -q
```

Measured:4 default32-head tokens pass8 state/output checks; the full Python
suite has684 passes and47 skips. The synthetic A7 chain passes416 acceptance
checks across96 NPU dispatches. These are synthetic component results; Track B
still needs whole-layer integration, xn buffering, segmented FFN and model tests.

## Production recurrent snapshots (Qwen3.8-27B)

For retained production captures, use `open_kernels/model/deltanet_state.py`:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/deltanet_state.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-entry/kernels \
  --out Models/qwen38-27b/head-state-validation/decode
```

Prepare captures with `make_decode.py --capture-boundaries` and a compatible
export. The production state has 48 heads, 130 padded rows per head for this
export, and three BF16 conv-history rows. Read geometry from the recipe; do not
reuse the standalone 128-row raster. The checker validates export hashes and
fixture identity, full sizes/finite values, zero initial state, padded zeros,
and bit-exact conv shifting. Its FP64 reference carries its own state across
positions, using captured records as inputs, never captured previous state.
It does not load model weights, rebuild kernels, or need NPU access.

Measured on 2026-10-13, build `63d7cd087942…`: 48 layers × 3 positions = 144
whole-state/output checks PASS. Worst state maxrel 1.76881057e-5, output
3.68898003e-5; all conv histories and padded zeros PASS. All 6912 head-state
diagnostics pass, but **99/6912 head-output diagnostics fail**. The worst head
has reference magnitude 9.91052341e-8, absolute error 4.89300530e-9, relative
error 0.0493718151 and cosine 0.9998768601 (L25/T0/H9). Maximum absolute error
among failed head outputs is 1.80884340e-7. Keep these failures visible; do not
relax thresholds or present the whole-tensor result as per-head acceptance.

Evidence: `Models/qwen38-27b/deltanet-state-validation/state-results.json`;
289 input hashes cover 144 activation/state pairs and the initial zero buffer.
The 288 captures match their previously retained hashes. Host TDD suite:
1168 passed, 47 skipped. No kernels or runtime arithmetic changed.
Record preparation (norm/projections, conv/L2, alpha/beta), composed state
accuracy and diagnosis of the head-output failures remain separate work.
