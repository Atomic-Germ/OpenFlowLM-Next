---
name: open-ffn-activation-carry
description: Build compensated wide FFN activation math with exact FP32 additions and compact down loops, then replay real rounding boundaries and the unchanged full model.
---

# Wide FFN activation rounding

Read `specs/open-engine/plans/qwen38-activation-precision.md` for acceptance
and limitations. Use the existing model and immutable `build_full_model`
references. This mode is experimental, not a runtime/catalogue promotion.

The source frame `build_norm_boundary/full/cold-0-layer4` has exact up/gate
at index14949 but its h rounds to the wrong BF16 neighbour. Run the existing
`diagnose-wide-ffn.py` trace replay first: baseline `build_down_precision/carry/ffn_trace`
has one activation mismatch, zero projection-induced BF16 mismatches.

`--activation-carry` requires a full FFN and product correction. It enables
nine-component compensated BF16 multiplication, exact integer-lane FP32
add/sub in the existing polynomial/Newton SiLU, and a compact core loop over
equal-width down segments. First-segment reset is computed inside the loop.
Host DMA order, weights, buffers and interfaces are unchanged. Default mode
keeps the original arithmetic and unrolled core schedule.

```bash
base=open_kernels/designs/wide_deltanet/build_activation_boundary/carry_add
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --ffn-correction --product-correction --block-carry --segment-carry \
  --activation-carry --trace --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --ffn-correction --product-correction --block-carry --segment-carry \
  --activation-carry --out "$base"
```

Build sequentially: layer_x shares generated translation units. Prepare and
compare both artifacts with `test-segmented-dense.py`; run `segmented.cfg`
using the existing harness with `HARNESS_TIMEOUT_MS=30000` and
`LD_LIBRARY_PATH=/opt/xilinx/xrt/lib`. Normal and traced activation arenas
must match bitwise for all13 inputs; the checks total26 and52 respectively.

Use `diagnose-wide-ffn.py --out <source> --tag cold-0-layer4 --trace-build
<new ffn_trace> --trace-out <fresh directory>`, run its `trace.cfg`, then
`--out <trace directory> --compare-trace --require-exact-h-bf16`.
Repeat for layers0 and1; layer0 additionally requires
`--require-exact-fo-channel 2931`. All three frames pass with this build.
Do not interpret `matches_recorded_h/fo: false` as tracing corruption after
changing arithmetic; use normal/trace equality and `repeat_exact` instead.

Keep the Horner polynomial loop rolled: unrolling six call sites overflows
program memory. All-IEEE integer arithmetic also failed the monolithic FFN
program limit and is not an available mode. Nine products alone fit after
rolling Horner but did not close14949; exact additions are also needed.
Final normal/trace text is16176/16320 B, plus24 B coefficient data. Total
allocated data plus stack is63576/64088 B. The trace leaves64 B program space.

For failed-build size diagnosis use `utilities/build-design-retain-failure.py
DESIGN OUT` with the same environment as the normal builder. It suppresses
the installed IRON cleanup hook only in that process and preserves the failing
exit code. This private hook is tied to the installed mlir-aie version. Read
`llvm-size -A` and `llvm-nm -S --size-sort`: ordinary llvm-size counts coefficient
data in its text total, which can mislead the16 KiB program check.

Rebuild without the activation flag and verify all13 old activation arenas
against `build_down_precision/carry/ffn`. CPU regression is
`ironvenv/bin/python -m pytest specs/open-engine/tests -q`.

Full replay: use original `build_full_model`, output projection
`build_ffn_boundary/carry/projection_k6144`, new normal FFN, LN
`build_norm_boundary/rne`, attention `build_model_precision/attention`, and
attention projection `build_attention_boundary/carry/projection_k5120`.
Run `decode.cfg`, `test-wide-full-model.py compare` and
`diagnose-wide-model-rounding.py`. Never change seeds, references or tolerances.

Recorded full result:3564 calls, tokens and reset pass;18895/18896 slice and
4121/4122 decode checks pass. Two failures remain, with worse final residual
and norm maxrel than before. Do not promote this mode. The first ten norms
are exact; layer5 xn has one local error at3295, reproduced in
`carry_add/layer5_norm`. That strict replay intentionally fails and is the next
starting point. Residual sums remain exact; local norm counts16 versus9 reflect
different device inputs and do not establish full-model accuracy.
