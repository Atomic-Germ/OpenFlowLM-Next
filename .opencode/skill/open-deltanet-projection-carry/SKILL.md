---
name: open-deltanet-projection-carry
description: Separate local and propagated real DeltaNet errors and validate compensated QKV/Z projections against unchanged 64-layer model references.
---

# Real DeltaNet QKV/Z projection carry

Read `specs/open-engine/plans/qwen38-deltanet-boundary.md`. Baseline after the
main merge is `build_main_oct3/full`, byte-identical to `build_activation749/full`.
Its first changed norm is cold-0-layer10/xm; xn is exact. The conditional
reference finds zero local BF16 conv-state or post errors: their differences
are propagated. Conv capture is shifted input state, not the convolution's
SiLU output. Records contain the actual normalized Q/K and activated V.

```bash
wide=open_kernels/designs/wide_deltanet
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-deltanet-boundary.py \
  --out "$wide/build_main_oct3/full" --tag cold-0-layer10
```

The utility checks fixture/kernel hashes, capture sizes and canaries. Norm
padding is intentionally poison and excluded only after validating the full
buffer. Logical recurrent state is decoded from padded production rows.
Conditional FP64 replays always retain `diagnostic_only: true, passed: false`.
Local/propagated counts overlap and must not be added. It never modifies inputs,
reference NPZs or device captures. Its exit0 means diagnosis completed, not
model acceptance.

Build serially with other layer_x builds (shared generated translation units):

```bash
base=open_kernels/designs/wide_deltanet/build_deltanet_boundary
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 5120 --projection-n 16384 --projection-correction \
  --product-correction --block-carry --out "$base/carry"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-dense-projection.py prepare \
  --build-dir "$base/carry/projection_k5120"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/carry/projection_k5120/projection.cfg"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-dense-projection.py compare \
  --build-dir "$base/carry/projection_k5120"
```

All14 inputs pass, including five cancellation cases; main-core text6288 B.
The N16384 output is concatenated QKV10240 then Z6144, using existing packing.
The arithmetic is the existing compensated product/block carry, now tested
on this role; no new runtime mode or family is introduced.

Full replay uses `utilities/replay-wide-model.py --source "$wide/build_full_model"`
with the output/attention/LN overrides from `open-ffn-sigmoid-series/SKILL.md`,
FFN `build_main_oct3/ffn`, plus
`--deltanet-projection "$base/carry/projection_k5120" --out "$base/full"`.
That option requires the exact K5120/N16384 geometry, passing primitive results
and unchanged hashes. Run decode.cfg on NPU, then test-wide-full-model.py compare,
diagnose-wide-model-rounding.py and the DeltaNet diagnostic. Do not regenerate
original references or promote based only on matching tokens. Logs use
`/tmp/dn-boundary-*.log`.

Candidate layer10: QKV has one FP32 difference (index1961, maxabs1.86265e-9),
Z and conv-state are exact. Gated output differs at604/head4 only; xm differs
at2315,2508,2814, all propagated. Conditional glue+step preserves the error;
reference records remove it. Device AB alone reproduces it, while device QKV
alone does not. Next isolate AB dot products versus nonlinearities at head4.
Use the report's exact values as diagnostic evidence, not a new acceptance
fixture or a reason to loosen full-model thresholds.

Full result:3564 dispatches complete, matching tokens and passing reset/state/
cache/canaries; all384 residual additions exact. Acceptance remains FAIL:
18893/18896 slice,4121/4122 decode (four failures versus six). Remaining gates
are layer47/head11_og, layer61/y (new), layer63/y and final_norm on cold-1.
Final residual maxrel0.020431852984951898 improves, final norm0.01282051282051282
is unchanged. First21 norm boundaries remain exact. Do not promote defaults
or call PR4 complete. The independent full-model comparator must retain exit1.
