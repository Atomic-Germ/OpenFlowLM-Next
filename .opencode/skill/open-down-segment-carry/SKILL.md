---
name: open-down-segment-carry
description: Build and validate compensated segmented Q4 down reduction, including real layer0 rounding and immutable full-model replay.
---

# Down segment residuals

Use the existing ironvenv, model and full fixtures. Read
`specs/open-engine/plans/qwen38-down-segment-precision.md` for measured results.
This is an opt-in precision probe; passing one frame does not accept PR4.

The real cold0/layer0 channel2931 loses precision both when each K4096 partial
is rounded and when five FP32 partials are added. Compensating only the latter
cannot recover the discarded residual. `--segment-carry` preserves the final
GEMV rounding residual in the existing table scratch, then accumulates high
and low components in two640-float planes of the existing1280-float ds buffer.
The table residual uses even/odd lane order; down accumulation must restore
row order. Both planes reset on the first segment of every frame. Do not
increase buffers or change packed weights to mask a numerical error.

Build sequentially (the generator shares translation units):

```bash
base=open_kernels/designs/wide_deltanet/build_down_precision/carry
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --ffn-correction --product-correction --block-carry --segment-carry \
  --trace --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --ffn-correction --product-correction --block-carry --segment-carry --out "$base"
```

Prepare and compare each primitive with
`utilities/test-segmented-dense.py prepare --build-dir <probe>` and
`utilities/test-segmented-dense.py compare --build-dir <probe>`.
Run its `segmented.cfg` with the persistent `open_kernels/harness/out/run_kernel`,
`HARNESS_TIMEOUT_MS=30000` and `LD_LIBRARY_PATH=/opt/xilinx/xrt/lib`.
Compare normal/traced h and fo bitwise, and check actual linker placements.
NPU runs are sequential. Omitting `--segment-carry` rebuilds the prior mode.

After the trace primitive passes, replay the unchanged real frame:

```bash
source=open_kernels/designs/wide_deltanet/build_ffn_boundary/carry/full
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-ffn.py \
  --out "$source" --tag cold-0-layer0 --trace-build "$base/ffn_trace" \
  --trace-out "$base/layer0_trace"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/layer0_trace/trace.cfg"
ironvenv/bin/python utilities/diagnose-wide-ffn.py --out "$base/layer0_trace" \
  --compare-trace --require-exact-h-bf16 --require-exact-fo-channel 2931
```

Use fresh trace/replay destinations. `--down-segments` on the original full
frame computes independent FP64 segment partials from packed bytes. Its
rounded-partial and sequential sums distinguish the two error sources;
these diagnostic arrays never replace acceptance references.

Replay all64 layers using `utilities/replay-wide-model.py`, the new normal FFN,
`build_ffn_boundary/carry/projection_k6144` and `build_model_precision/ln`.
Keep other kernels, source fixtures, seeds and bounds unchanged. Compare with
`utilities/test-wide-full-model.py compare`; then locate the first differing
norm using `utilities/diagnose-wide-model-rounding.py`. Report full failures
regardless of the isolated frame result. Default runtime/catalogue support
remains gated on full-model acceptance.

Recorded result: the strict real channel passes; normal/traced FFN pass26/52
checks, and both variants agree bitwise on13 inputs. The prior mode's13
captures are unchanged. The trace text occupies exactly16384 bytes; normal
text16240 bytes, so check the program budget before extending this kernel.
Full replay still fails four numerical checks. Both norms in cold0/layers0-2
are exact; next inspect layer1 residual addition at3390, where captured res1
and fo match but device y differs from their CPU FP32 sum. This propagates
into layer3 entry norm. See the report for exact values and unchanged bounds.
