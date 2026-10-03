---
name: open-wide-ab-carry
description: Build compensated banked BF16 AB projections, isolate real dot versus nonlinear errors, and replay the unchanged 64-layer model.
---

# Wide AB projection carry

Read `specs/open-engine/plans/qwen38-ab-precision.md`. Baseline is
`build_deltanet_boundary/full`, not the earlier sigmoid-only trajectory.
At cold-0-layer10/head4, beta is correct given the device beta-logit, but that
logit differs from the independent projection. Keep the original references.

Use the installed ironvenv (Python3.14, mlir-aie1.4.2, llvm-aie21) and XRT.
Unlike layer_x builds, this design does not rewrite shared generated sources.

```bash
base=open_kernels/designs/wide_deltanet/build_ab_precision
for width in 5120 2560; do
  PATH=/opt/xilinx/xrt/bin:$PATH WIDE_AB_CARRY=1 WIDE_DN_HIDDEN=$width \
    ironvenv/bin/python open_kernels/build_design.py \
    open_kernels/designs/wide_deltanet/ab.py "$base/carry$width"
  ironvenv/bin/python utilities/test-wide-deltanet-ab.py prepare \
    --hidden "$width" --build-dir "$base/carry$width"
  HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
    open_kernels/harness/out/run_kernel "$base/carry$width/ab.cfg"
  ironvenv/bin/python utilities/test-wide-deltanet-ab.py compare \
    --hidden "$width" --build-dir "$base/carry$width"
done
```

Both widths pass28 primitive checks over seven dispatches. With WIDE_AB_CARRY=0,
rebuild H5120 into a separate directory and compare all seven results to
`build_ab_h5120`; the recorded outputs are byte-identical. Never overwrite that
baseline's accepted binaries. The switch defaults off; no catalogue promotion.

The new `ab_carry.cpp` reuses q4_product_add/TwoSum for exact BF16 products in
the normal range, accumulating a low FP32 component across all64-row tiles and
input chunks. Two private accumulators expand from32 to64 floats. Each bank's
first tile resets both components; finishing folds them before the unchanged
scalar NPU nonlinearities. This does not move neural math to the host. Packing,
three BOs, bank-tail behavior and DMA streams remain unchanged.

Replay the real frame with the accepted baseline kernel first, then candidate:

```bash
wide=open_kernels/designs/wide_deltanet
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-ab.py prepare \
  --source "$wide/build_deltanet_boundary/full" --kernel "$base/carry5120" \
  --out "$base/layer10" --tag cold-0-layer10 --head 4
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/layer10/ab-replay.cfg"
ironvenv/bin/python utilities/diagnose-wide-ab.py compare --out "$base/layer10" --require-exact-dots
```

Prepare only fresh directories. The tool validates source/kernel hashes,
extracts banked constants with production byte adapters and captures logical
xn while retaining poison padding. Two guarded runs must match. The strict
flag checks all96 FP32 dot outputs, not nonlinear acceptance. Results always
retain diagnostic_only=true and model_passed=false. Conditional nonlinear
references use rounded captured dots; original AB reference retains FP64 dots.
Do not conflate the two when a nonlinear result straddles a rounding boundary.

Full replay: use `utilities/replay-wide-model.py` with original `build_full_model`,
the complete overrides from `open-deltanet-projection-carry/SKILL.md`, and add
`--deltanet-ab "$base/carry5120" --out "$base/full"`. Geometry, primitive status
and hashes are validated before changing the d_ab binding. Run decode.cfg on
NPU, then test-wide-full-model.py compare, diagnose-wide-model-rounding.py and
selected diagnose-wide-deltanet-boundary.py frames. The full numerical gate
is separate from dot exactness or token agreement. Logs: `/tmp/ab-precision-*.log`.

Measured text:14800/11392 B at H5120/H2560; placed data plus reserved4096-byte
stack:21760 B. The old path uses21504 B. Real repeated H5120 call in this short
run is1.422 ms versus0.560 ms baseline; this is not an application benchmark.

The layer10 dot gate closes, but gated output604 remains different. With the
new AB, conditional post on captured O/Z now matches the independent BF16
reference; the device's remaining mismatch is local to post. All AB/QKV
counterfactuals produce correct rounding on this frame. Next isolate head4/
lane92 post normalization and gate arithmetic. Conditional FP32 is
0.0002737044997047633, reference BF16 0.0002727508544921875, device BF16
0.000274658203125. See the retained boundary/summary JSONs under `full`.

Full acceptance still fails four checks (18893/18896 slice,4121/4122 decode):
cold-1 layer47 heads1 and22, final layer63 residual and final_norm. Head1 fails
cosine despite passing maxrel. Prior head11/layer61 failures disappear.
Final residual maxrel0.01929894806260552 and final norm0.008547008547008548
improve but exceed unchanged0.005/0.008 bounds. Tokens/reset/state/cache pass,
all384 residual sums remain exact, and first21 norm boundaries remain exact.
PR4 is incomplete; retain the experimental switch and comparator exit1.
