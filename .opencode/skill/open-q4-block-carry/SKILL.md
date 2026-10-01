---
name: open-q4-block-carry
description: Diagnose real wide FFN rounding with up/gate traces and validate Q4 block-residual carry on NPU and unchanged full-model fixtures.
---

# Real FFN boundaries and Q4 block carry

Use `ironvenv` and the existing model fixtures. Read
`specs/open-engine/plans/qwen38-ffn-boundary.md` for acceptance and limitations.
This mode is an opt-in probe; never infer runtime support from a local gate.

`utilities/diagnose-wide-ffn.py --out <full replay> --tag cold-0-layer1`
separates the error in FFN activation from the down projection, using captured
BF16 xm and h from the activation arena. It verifies selected fixture/kernel
hashes and capture guards. Conditional references are diagnostic output;
the original model trajectory and acceptance references are never replaced.

Build sequentially because layer_x shares generated translation units:

```bash
base=open_kernels/designs/wide_deltanet/build_ffn_boundary/carry
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --ffn-correction --product-correction --block-carry --trace --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --ffn-correction --product-correction --block-carry --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 6144 --projection-n 5120 --projection-correction \
  --product-correction --block-carry --out "$base"
```

The new flag requires product correction. Block carry retains both components
through the global sum and renormalizes after every K32 block. The table,
per-partition reset, weight format and down segmentation remain unchanged.
An unnormalized sum carried across the entire K dimension failed an earlier
cancellation regression; do not reintroduce it.
To rebuild the failing product-only baseline, omit `--block-carry` and select
a separate output directory.

Prepare/run/compare the standard FFN, traced FFN and projection fixtures via
`test-segmented-dense.py` and `test-dense-projection.py`. The product projection
fixture automatically includes the cancellation cases. Check normal/trace h
and fo for bit equality on the13 standard inputs before interpreting a trace.
Use `HARNESS_TIMEOUT_MS=30000`, `LD_LIBRARY_PATH=/opt/xilinx/xrt/lib` and the
persistent `open_kernels/harness/out/run_kernel`. NPU runs are sequential.

Once the trace primitive passes, reproduce the real layer1 test:

```bash
source=open_kernels/designs/wide_deltanet/build_product_precision_v2/full_ffn_ln
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-ffn.py \
  --out "$source" --tag cold-0-layer1 --trace-build "$base/ffn_trace" \
  --trace-out "$base/layer1_trace"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/layer1_trace/trace.cfg"
ironvenv/bin/python utilities/diagnose-wide-ffn.py --out "$base/layer1_trace" \
  --compare-trace --require-exact-h-bf16
```

Trace destinations must be fresh. The isolated FFN reserves438272 activation
bytes versus397312 in the model harness; the utility preserves logical bytes,
adds poisoned padding and moves the canary. It copies diagnostic references
and hashes all replay inputs, linking immutable pool/kernel files.

The old product mode has six h BF16 mismatches on this frame, all attributable
to up/gate projections, with none added by activation math. Block carry closes
that strict gate. `matches_recorded_h/fo` compare to the old full-run captures;
false is expected for a changed arithmetic mode and does not itself imply that
tracing perturbs computation. `repeat_exact` must remain true.

Then use `replay-wide-model.py` with the new output and normal FFN, plus
`--ln open_kernels/designs/wide_deltanet/build_model_precision/ln`. Keep full
references and thresholds unchanged. Record actual full-depth failures even
when local rounding, token feedback and reset checks pass.

Recorded result: all primitive tests and the strict real layer1 h gate pass,
but the full replay still fails five numerical checks (four slice, one decode).
The next first-divergence target is layer0 down at channel2931: its h BF16,
projout and res1 are exact, but fo differs by-1.3969838619e-9 and flips the next
norm's rounding. Isolate the down segments and final FP32 accumulation. Layer2
entry norm is now exact. This is not an
accepted full-model replacement or a reason to widen any tolerance.
