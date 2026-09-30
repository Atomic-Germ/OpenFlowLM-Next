---
name: open-q4-product-precision
description: Reproduce the Q4 block cancellation regression and build opt-in compensated products for wide projection and FFN probes.
---

# Q4 product precision

Use `ironvenv` and the existing packed model and immutable full-model fixtures.
Read `specs/open-engine/plans/qwen38-q4-product-precision.md` for measured gates.
This is an isolated precision probe, not a supported runtime kernel family.

Build sequentially because layer_x shares generated translation units:

```bash
base=open_kernels/designs/wide_deltanet/build_product_precision_v2
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 6144 --projection-n 5120 --projection-correction \
  --product-correction --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --ffn-correction --product-correction --out "$base"
```

The product flag requires corrected activation tables. It changes neither
their allocation nor the packed weight format. Keep coarse and residual
integer dots separate through multiplication, compensate the block's products
before merging the result into the existing compensated K reduction, and
retain low bits in the activation block-sum tree. The FFN still segments down
at4096; do not increase that limit without checking actual L1 placement.

Prepare and run the standard projection and FFN fixtures using
`utilities/test-dense-projection.py` and `utilities/test-segmented-dense.py`.
Product-enabled projection metadata automatically adds five cancellation
cases to the original nine; the replay gate therefore includes this regression.
To isolate cancellation or reproduce the old kernel's failure, use a separate
cancellation directory: copy `final.xclbin`, `insts.bin` and
`probe-toolchain.json` from the projection build, then run:

```bash
ironvenv/bin/python utilities/test-dense-projection.py prepare \
  --cancellation --build-dir "$base/cancellation"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/cancellation/projection.cfg"
ironvenv/bin/python utilities/test-dense-projection.py compare \
  --build-dir "$base/cancellation"
```

The analytic test uses W=[0,0,-1,...], x=[1,2^-24,2^-24,0,...] per K32
block, including sign reversal, scaling, zero and repeat. At K6144 the result
is exactly -192*2^-24 for unit scale. Original corrected arithmetic returns
zero. Random inputs alone do not cover this cancellation failure.

Only after both ordinary and cancellation primitive gates pass, replay the
unchanged64-layer fixtures with `utilities/replay-wide-model.py`; see
`open-wide-model-precision/SKILL.md` for flags and comparison. Use fresh
destinations, preserve failed captures and never replace independent references.
Run the CPU suite and record resource use and full-model failures honestly.

The first output+FFN replay has a locally incorrect cold0/layer1/xn at
channel1914 (-0.2080078125 rather than -0.208984375); its conditional and
independent reference agree. Adding the existing compensated LN removes all
norm differences in layers0 and1, moving the first changed boundary to layer2
input (19 propagated differences). Follow the report for full-depth results;
do not infer model acceptance from this local improvement.
All three full-model replays still fail (3,5,3 numerical checks respectively
for output, output+FFN, output+FFN+LN). Tokens and reset replay match. The next
diagnostic target is the layer1 residual/FFN output feeding layer2.
