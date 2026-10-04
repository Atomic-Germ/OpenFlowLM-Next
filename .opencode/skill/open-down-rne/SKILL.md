---
name: open-down-rne
description: Diagnose propagated residual rounding, build exact final down rounding, and validate real FFN boundaries against unchanged model fixtures.
---

# Final down rounding and residual diagnosis

Read [the report](../../../specs/open-engine/plans/qwen38-residual-boundary.md).
Baseline is `build_post_precision/full`. At layer11/xm[4872], conditional norm
is exact and only the input residual reproduces the BF16 mismatch. The first
scalar error is layer0 down4872, despite exact BF16 activation. Do not change
RMSNorm or layer11 output projection based only on that later norm mismatch.

Use ironvenv (Python3.14, mlir-aie1.4.2, llvm-aie21) and XRT. Keep fresh output
directories; builds must be sequential because layer_x rewrites shared generated
translation units. Retain failed artifacts with OFLM_KEEP_FAILED=1 and inspect
per-core ELF text against16384 B. Larger data/stack budgets cannot fix text overflow.

```bash
wide=open_kernels/designs/wide_deltanet
base=$wide/build_residual_precision/final_rne
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-residual.py \
  --out "$wide/build_post_precision/full" --tag cold-0-layer11 --channel 4872
OFLM_KEEP_FAILED=1 ironvenv/bin/python utilities/probe-qwen35-wide.py \
  --scope ffn --ffn 17408 --ffn-correction --product-correction --block-carry \
  --segment-carry --activation-carry --activation-series --down-rne --trace --out "$base"
```

The residual diagnostic validates selected reference/kernel hashes and guarded
captures. Its operand and single-channel substitutions are diagnostic_only;
they never overwrite acceptance references. Scalar history records input and
operator errors separately from correctly rounded residual addition.

For a successful build, prepare/compare with `utilities/test-segmented-dense.py`
using `--build-dir "$base/ffn_trace"` and run its segmented.cfg through
`open_kernels/harness/out/run_kernel` with HARNESS_TIMEOUT_MS=30000 and
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib. Preserve the original13 primitive inputs.
Then replay the real frame:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-ffn.py \
  --out "$wide/build_post_precision/full" --tag cold-0-layer0 \
  --trace-build "$base/ffn_trace" --trace-out "$base/layer0"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/layer0/trace.cfg"
ironvenv/bin/python utilities/diagnose-wide-ffn.py --out "$base/layer0" \
  --compare-trace --require-exact-h-bf16 --require-exact-fo-channel 2743 \
  --require-exact-fo-channel 2931
```

The old accepted trace (`build_activation749/compact/ffn_trace`) reproduces
layer0 captured h/fo and fails4872. The retained --down-rne mode changes only
final high+low rounding, fixing2743 and reducing four FP32 differences to three.
The strict4872 gate still fails; run it separately and preserve its failure.

Experimental exact segment reducers exceeded the16384-byte instruction budget:
ordered all-integer operations16800 B, native exact residual subtractions16624 B.
Applying minsize to the helper did not reduce the latter. Their retained failed
ELFs live under sorted_rne, compact_rne and size_rne. These unbuildable reducers
are not part of the retained mode. Their normal-range CPU arithmetic tests did
pass; that does not establish hardware acceptance. The next target requires
isolating per-segment residuals and making instruction space before extending
compensation. Keep the existing high/low scratch, lane order and reset behavior.

For full regression of the final-rounding correction, build the normal FFN without --trace,
validate all13 inputs and normal/trace equality, then replay the immutable full
model with the complete overrides from open-wide-post-carry/SKILL.md, changing
only --ffn. Matching one channel is not model acceptance. Record actual full
failures and changed boundaries; PR4 remains gated on the original tolerances.

Use `build_residual_precision/full_down_rne` for this full replay; the `full`
directory under the same parent is an older experiment and must not be reused.
Normal/traced final-adder kernels pass26/52 checks and match all13 arenas.
Default rebuild matches build_main_oct3/ffn bytewise. Text is15424/15552 B;
measure .text with llvm-size -A, since plain llvm-size also counts40 B of
coefficient data in its combined column. Data plus reserved stack is
63592/64104 B. The correction remains an opt-in diagnostic;4872 is unresolved.

Full result:3564 dispatches complete,18892/18896 slice checks and4122/4122 decode
checks pass. Combined acceptance still fails. All384 xn/xm buffers and the decode
result JSON match the preceding post-carry run; the same four slice checks fail.
No full-depth numerical gate closes. See the report for unchanged max errors.
Next build a down-only probe to capture segment high/low values at4872 without
linking SiLU, then determine whether exact segment reduction fixes that channel
before making space in the full FFN. Do not promote the mode or widen tolerances.
