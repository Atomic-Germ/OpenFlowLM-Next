# CPU prefill projection rounding attribution

Analyze the retained L34/P17, L37/P7 and L53/P17 prefixes without a native build
or NPU run. Revalidate the capture audit, native/compiler/kernel provenance,
independent references and exact baseline logits/residuals. The two hybrid
schedules have identical captures; use one representative. Only one layer's
weights are decoded at a time. This tool does not measure sequential projections.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-projection-attribution/check.py \
  --run Models/qwen38-27b/prefill-prefix-sublayers/run \
  --build Models/qwen38-27b/prefill-prefix-sublayers/build \
  --baseline Models/qwen38-27b/prefill-weighted-cli/v2/candidate/run \
  --out Models/qwen38-27b/prefill-projection-attribution/run
```

Always use a fresh output directory. Failures retain an incomplete report.
Successful exit means that the diagnostic completed, not model acceptance.
`model_acceptance` is always false. Weights/captures and primary references
are unchanged; artifacts stay under ignored `Models/`.

## Projection prediction

Reconstruct QKV/Z/out packed weights using the manifest's `std_perm` operations,
dequantize the RS2 layout, and require exact equality to source effective weights.
Record hashes of both packed and effective values. This establishes their
agreement under the current packer, not a fresh device-side weight readback.

For the actual captured FP32 input `x`, compute FP64 matrix products with:

1. Effective weights `W` and original `x` (`exact`).
2. `W` and `BF16(x) + BF16(x - BF16(x))` (`paired`).
3. `BF16(W)` and the same paired input (`rounded`).

Compare actual NPU output to these predictions. The difference between `paired`
and `exact` isolates input approximation; `rounded - paired` predicts weight
rounding. `native - rounded` also includes finite-precision accumulation and
high/low summation. The CPU predictor does **not** emulate NPU accumulation or
claim byte-exact replay. A close fit supports the rounding hypothesis; a new
precision kernel still requires an isolated byte-exact native control.

Score Q, K, V and Z **per position and per head**, each against its own exact
reference scale; output projection is one hidden vector at the selected row.
There are 7683 cases across the three layers. Preserve every failing head gate,
including failures hidden by the previous aggregate QKV scale. All original
maxrel bounds stay <0.005. The L2 remaining-error ratio compares prediction gap
to native-minus-exact error; undefined ratios are null, never NaN/infinity.

## Causal substitutions

Start a fresh independent host state for each variant and replay every row
through the selected position. Never read future positions. Replace captured
Q, K, V, Z or QKV projections with effective-weight results on **captured norm**,
preserving the FP32 projection-storage boundary. Recompute the downstream
DeltaNet and out projection in FP64. These are ideal-suffix counterfactuals,
not corrected NPU executions; upstream inputs remain native and drifted.

Retain signed residual components on the **same independent pre-FFN reference
scale**: propagated input, QKV, Z, their nonlinear interaction, host arithmetic,
out projection, residual addition. Verify vector closure with a floating-point
rounding bound. Individual Q/K/V substitutions are non-additive diagnostics.
Scalar maxima of different components cannot be summed. NPZ files save the
predictions, native arrays, variants, signed components and reference vectors.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q \
  specs/open-engine/tests/test_prefill_projection_attribution.py
```

Tests cover paired input versus weight rounding, exact head coverage and small
head errors hidden by other heads, non-finite inputs, zero errors, causal
prefix/reset behavior, signed cancellation/interaction and failure persistence.

## Validated checkpoint (2026-10-21)

Nine manifest-packed/source effective-weight comparisons are exact. Provenance
and all 81 prior captures validate. All 7683 comparisons complete; **7681 pass
the native-versus-effective-weight maxrel gate**. Two failures were hidden by
aggregate QKV scaling (indices are zero-based):

| Role | Layer / position / head | Native maxrel | BF16-weight prediction gap maxrel |
|---|---|---:|---:|
| Q | 34 / 6 / 13 | 0.0050724904 | 7.3845882e-7 |
| V | 37 / 0 / 46 | 0.0071224278 | 1.4109131e-6 |

Worst prediction gap across every role/head is **2.9830263e-6**. The remaining
L2 error ratio is at most **0.0022613141** (0.2262% of native-minus-exact error);
this strongly supports BF16 weight rounding as the source of these local
projection differences. The prediction includes paired-input approximation,
whose own worst maxrel is 1.3430085e-5. This is not byte-exact native control.

All three signed residual decompositions close (max discrepancy 2.2204460e-16).

| Selected pre-FFN row | Native maxrel | Joint ideal QKV/Z/host/out maxrel | QKV component | Z component | Out-projection component |
|---|---:|---:|---:|---:|---:|
| L34/P17 | 0.0007177295 | 0.0006092193 | 0.0001205549 | 0.0000291166 | 0.0000179964 |
| L37/P7 | 0.0010862168 | 0.0008486373 | 0.0001993436 | 0.0000170513 | 0.0000323107 |
| L53/P17 | 0.0075342533 | 0.0066083261 | 0.0001410142 | 0.0001375375 | 0.0008264166 |

At L53/P17 even the joint ideal suffix remains above 0.005, so no local
correction here removes inherited drift. At that row out-projection rounding
is the largest measured local residual component. Components use the same
reference scale but their maxima are not additive.

Next replay the fused entry GEMM **N16384/K5120/T256** on captured L34/P6 and
L37/P0 (norm rows from the L34/P17 and L37/P7 prefix files). Require native
paired-input output byte-exact to concatenated QKV/Z captures, then test
paired-weight precision with unchanged per-head gates and two slot placements.
This directly targets the two newly exposed failures without loading all model
weights. Retain L53/P17 out projection as a subsequent precision target; neither
experiment alone proves long-prefix/model acceptance.

Eleven initial RED tests then pass; full host suite **1592 passed, 47 skipped**.
No native rebuild or NPU dispatch. Full retained residual accuracy remains
**3974/4032 FAIL**, and primary references are unchanged. Evidence is under
`Models/qwen38-27b/prefill-projection-attribution/`.
