---
name: open-ffn-sigmoid-series
description: Trace real compensated SiLU stages, build the small-argument sigmoid series within wide FFN program limits, and validate real boundaries and immutable full-model fixtures.
---

# Wide FFN small-argument sigmoid

Read `specs/open-engine/plans/qwen38-sigmoid-precision.md` for the evidence and
recorded acceptance. The RED frame is `build_norm_finish/layer8_ffn`, index749:
one BF16 activation error despite exact up/gate. Original full-model references
live in `build_full_model`; don't regenerate them or relax gates.

For isolated diagnosis build `open_kernels/designs/layer_x/activation_probe.py`
with `open_kernels/build_design.py DESIGN OUT`. Set `PROBE_ACTIVATION_SERIES=1`
for the candidate; omit it for the original compensated path. The probe calls
the same C++ helpers as production. Do not use `-Oz`: the integer adder grows.

```bash
wide=open_kernels/designs/wide_deltanet
ironvenv/bin/python utilities/trace-wide-activation.py prepare \
  --source "$wide/build_norm_finish/layer8_ffn" --kernel <probe-build> \
  --out <fresh-frame-directory> --index 749
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel <fresh-frame-directory>/trace.cfg
ironvenv/bin/python utilities/trace-wide-activation.py compare --out <fresh-frame-directory>
```

The diagnostic deliberately exits1 and retains `passed: false`, even when
BF16 matches. It reports ten FP32 stage planes and a separate production h.
The first nine planes describe the original exp/Newton path; only production h
uses the selected series mode. Check repeat, canaries and baseline agreement
with recorded FFN h. Channel893 supplies a real input outside the interval.

Build full FFN sequentially because generators share translation units:

```bash
base=open_kernels/designs/wide_deltanet/build_activation749/compact
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --ffn-correction --product-correction --block-carry --segment-carry \
  --activation-carry --activation-series --trace --out "$base"
# Repeat without --trace for the normal FFN.
```

Series requires activation carry. It applies an odd degree9 polynomial only
for |gate|<=0.5 and preserves the exp/Newton branch elsewhere. Speculative
out-of-range polynomial lanes are zeroed before evaluation. It is not globally
correctly-rounded sigmoid. Compact up/gate and dynamic full/tail down loops
are enabled only for this mode; host DMA and accumulation order are preserved.

Use `test-segmented-dense.py prepare/compare --build-dir <ffn or ffn_trace>`
and run `segmented.cfg`. Compare normal/trace arenas bytewise for all13 inputs.
Replay layers0,1,4 and8 with `diagnose-wide-ffn.py --trace-build <ffn_trace>`;
run `trace.cfg`, then `--compare-trace --require-exact-h-bf16`. Layer0 also
requires `--require-exact-fo-channel 2931`. Rebuild without the series flag and
compare against `build_main_merge/ffn` to check previous-mode preservation.

Failed `series` and `dynamic` builds exceed program memory. Use
`utilities/build-design-retain-failure.py` with the same build environment to
retain failed ELF files; inspect `llvm-size -A`. Don't count coefficient data
as program text or assume a successful compile establishes hardware acceptance.

Full replay uses `utilities/replay-wide-model.py` with original `build_full_model`,
new normal FFN, LN `build_norm_finish/carry`, output projection
`build_ffn_boundary/carry/projection_k6144`, attention `build_model_precision/attention`
and attention projection `build_attention_boundary/carry/projection_k5120`.
Run `decode.cfg` and both `test-wide-full-model.py compare` and
`diagnose-wide-model-rounding.py` with `OPENBLAS_NUM_THREADS=1`. Keep all failures
visible; matching tokens alone never closes PR4.

Recorded primitive results:26/26 normal and52/52 trace checks pass; all13
normal/trace arenas are byte-identical. Layers0,1,4,8 h BF16 are exact, layer0
down2931 remains exact, and all17 layer8 gates outside the interval retain
the old FP32 h. Rebuilt previous mode matches all13 old arenas. Final normal/
trace text is15440/15568 B, data plus stack63592/64104 B; both instruction
streams are unchanged. CPU suite:854 passed,47 skipped. These local results
are separate from full-model acceptance.

Full result (`build_activation749/full`): all 3564 dispatches complete and
tokens/reset/state/cache/canaries pass, but numerical acceptance worsens to
six failures from two (18891/18896 slice, 4121/4122 decode). Four new layer47
attention-head failures accompany final residual maxrel 0.026122822719335435
and unchanged final norm maxrel 0.01282051282051282. Do not promote this mode
or report PR4 complete. All 384 residual additions remain exact.

First 21 norm boundaries are exact; the next is layer10/xm with seven
propagated differences and no local norm error. Investigate layer10 DeltaNet
projection/conv/post from its exact xn next. Conv has two BF16 differences,
gated output six; these independent-path comparisons alone do not identify
the first faulty primitive. See `full/layer10-boundary-summary.json` and the
linked report for all full-model failures and unchanged bounds.
