# Paired A/B of one arithmetic change

`ab.py` answers "did this change help?" when the only thing that visibly moved is the
number of positions failing `corr > 0.9999`.

That count is a threshold, not a measurement. On data whose defect spans five decades
(the body sits near `1e-6`, single positions reach `3e-1`), a change can move the count
in either direction while having no effect at all. Reading the count as a verdict is how
a correct kernel change gets reverted and how a wrong one gets shipped.

## What it computes

| field | question it answers |
|---|---|
| `paired_*`, `t`, `direction` | is the change systematic, or a coin flip? Paired t on the per-position defect difference. |
| `flips[].material` | did the position land clearly on the far side of the gate, or scrape past it? |
| `component(...)` | does the changed component add to the total error, or cancel part of it? |

`material` uses `margin_over_gate`: how far onto the far side a flip landed, as a
multiple of the gate defect. The default factor is `2.0`. Crossing a threshold always
moves further than the distance to it, so "did the change cause the flip" is uselessly
always yes; the materiality margin is the question with information in it.

`component(total, share)` reports `cosine` and a `role`: `adding` (cos > 0.1),
`neutral`, or `cancelling` (cos < -0.1). A cancelling component is never the first thing
to fix -- removing it makes the total worse, which looks like a regression.

## Measured case: the norm precision fix (2026-10-31)

`vecmath.h`'s `splitN` reaches ~16 mantissa bits; `mac_vv`/`mac_vs` split an fp32 operand
into two bf16 terms, which is a relative `2e-5` on a 5120-wide row. The norms
(`ln.cc`, `ln_xn.cc`, `ln_nr32.cc`) were moved to the three-term `mul3bv_32`/`mul3_32`.
`xn` fell `8.450e-06 -> 2.082e-07` in the full model. The sequential route's failing count
then rose `66 -> 75`, which looked like a regression.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-long-context/ab.py \
  --before  Models/qwen38-27b/prefill-long/trace-513 \
  --capture Models/qwen38-27b/prefill-long/trace-513-v3 \
  --reference Models/qwen38-27b/prefill-long/boundaries-513/513 \
  --positions 513 \
  --out Models/qwen38-27b/prefill-long/ab-norm-fix-v1.json
```

```
improved 217  worse 296
paired mean change -1.616e-05 (t = -0.31) -> inconclusive
gate flips 25, of which 20 are not material (within 2.0x of the gate)
```

20 of 25 flips had a median margin of `1.29x` over the gate; only 5 were material (three
`now_passing` at `2.2x`, `2.3x`, `10.3x`; two `now_failing` at `2.4x`, `4.7x`). The
`66 -> 75` move is threshold noise: those positions sit a few percent from the gate and
the defect's own scatter is larger than that gap.

Why the count moved at all -- signed decomposition at L0/P0, against the FP64 reference:

| quantity | before | after |
|---|---|---|
| `xn` maxrel | `8.450e-06` | `2.082e-07` |
| `Wz @ xn_error` (the norm's share of the `z` error) | `8.789e-04` | `3.188e-05` |
| total `z` error | `3.009e-03` | `2.899e-03` |
| `cosine(norm share, total)` | `0.291` | `0.098` |

The norm's error was a small, partly **anti-correlated** part of a total dominated by the
per-token projection arithmetic. Removing it cut its own share `27.6x` and the total by
`3.7 %`, because most of what it removed had been cancelling the larger term. That is why
`max` defect fell `4.71e-02 -> 2.32e-02` while `p99` rose `3.16e-03 -> 4.37e-03`: the old
`p99` was artificially low at the positions where cancellation happened to be favourable.

Verdict: the norm fix is correct and its effect on this route is **neutral within
scatter**. No tolerance was changed and no improvement is claimed.

## Scope

This is a diagnosis, never an acceptance row: `model_acceptance` is always `false`, and
the tool refuses a position it cannot score rather than skipping it. Nothing here
replaces the gate -- it explains what moving the gate's count means.

Two things it will not tell you, by construction: whether the two captures came from the
same kernel bundle (it compares captures, not bundles), and whether the reference covers
these positions (it truncates the correlation window to the shorter vector and refuses a
window below two logits, so a reference prepared for fewer positions is caught as a
mismatch rather than scored on the overlap).
