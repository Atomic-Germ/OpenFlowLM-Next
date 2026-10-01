---
name: open-residual-rne
description: Build and validate exact vector FP32 residual addition in wide RMSNorm, including rounding boundaries, signed zero and unchanged full-model replay.
---

# Wide residual round-to-nearest

Read `specs/open-engine/plans/qwen38-residual-precision.md` for the measured
limits. Use the existing ironvenv and model fixtures. This is an opt-in probe,
not an accepted runtime kernel set.

The preceding segmented-down stage exposed matching layer1 res1/fo inputs at
channel3390 whose device sum differs by one FP32 step. Baseline random LN cases
also have hundreds of non-exact residual sums. `LN_RESIDUAL_RNE=1` uses the
integer-lane algorithm in `include/fp32_add_rne.h`, retaining guard/round/sticky
bits and rounding ties to even. It does not replace global vecmath additions.
The RMSNorm finish preserves signed residual zero, including weight sign.

```bash
base=open_kernels/designs/wide_deltanet/build_residual_precision
LN_N=5120 LN_STREAM_COMPENSATED=1 LN_RESIDUAL_RNE=1 PATH=/opt/xilinx/xrt/bin:$PATH \
  ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/ln/ln.py "$base/ln_rne"
ironvenv/bin/python utilities/test-wide-ln.py prepare \
  --build-dir "$base/ln_rne" --exact-residual
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/ln_rne/ln.cfg"
ironvenv/bin/python utilities/test-wide-ln.py compare --build-dir "$base/ln_rne"
```

The preparation flag adds six cases to the original12 and records strict
residual-bit acceptance in the fixture. The norm thresholds remain unchanged.
Repeat and canaries remain mandatory. Normal random data, the captured real
pair, signed zeros, cancellation, ties and varied exponent gaps are exercised.
Do not overwrite accepted fixture directories; use fresh build destinations.
To reproduce RED, link the immutable `build_model_precision/ln` artifacts into
a fresh directory and prepare the same strict fixture. To check legacy mode,
rebuild with `LN_RESIDUAL_RNE=0`, prepare without `--exact-residual` and compare
all12 y/xn captures bitwise with `build_model_precision/ln`.

Run `ironvenv/bin/python -m pytest specs/open-engine/tests -q`. The integer
adder test compiles the actual header with a host C++ compiler, then checks
200000 random bit pairs and289 Cartesian boundary pairs against NumPy FP32.
It includes subnormals, infinities and NaN classification; the NPU norm test
covers finite model-range inputs, not normalization of nonfinite values.
The host routine is a test, never an inference fallback.

Replay all64 layers with `utilities/replay-wide-model.py`, using the original
`build_full_model` source, `build_ffn_boundary/carry/projection_k6144`,
`build_down_precision/carry/ffn`, and the new `ln_rne`. Use a fresh destination,
run its `decode.cfg`, then `test-wide-full-model.py compare` and
`diagnose-wide-model-rounding.py`. References, seeds, weights and tolerances
must stay unchanged. Report remaining numerical failures even when tokens,
reset and local residual checks pass.

Resources for the accepted primitive build: text6736 B, data plus reserved
stack57600 B. DMA instructions are unchanged. A short dispatch sample was
slower than the old mode; this is accuracy work, not a performance benchmark.

Recorded full result:0 residual-bit differences over384 additions (1966080
values), versus206956 in the previous replay. All3564 calls, three tokens and
reset succeed, but11 numerical gates fail versus4 before; do not promote this
mode. First divergence is layer3 post norm (72 propagated BF16 differences),
with layer3 entry exact. The first-token attention output has27 differing BF16
values: a conditional single-row V*sigmoid(gate) check has21 device differences
and6 oracle differences. Isolate projected gate/V rounding and attention math
next; preserve the full-model acceptance references.
