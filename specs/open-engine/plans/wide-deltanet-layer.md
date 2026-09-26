# Wide DeltaNet complete-layer bring-up — 2026-09-25

**Historical report; the gate is closed by the [2026-09-26 precision follow-up](wide-deltanet-precision.md).**
The results below describe the original failure. A synthetic all-Q4
H5120/FF17408 layer now executes entirely on the open NPU path, but one of
eight outputs exceeds the inherited full-layer error bound. This follows
[wide attention](wide-attention.md). No catalogue, converter or model support
is promoted. The follow-up records the fix and current next step.

## Composition and TDD

`recipes/wide_deltanet_layer.py` composes ten existing dispatches per token:
entry residual RMSNorm; QKV+Z projection; banked AB; convolution/records;
48-head recurrence; head RMSNorm/SiLU gate; output projection; post residual
RMSNorm; segmented FFN; final residual add using the LN primitive. The final
unused normalization output is discarded. There is no CPU neural computation
between dispatches, only harness BO copies/synchronization.

The isolated Q4 projection accepts a tiled output width: K5120/N16384 combines
10240 QKV and6144 Z rows; K6144/N5120 produces the mixer output. Both retain
actual main-core scratch and production prep/GEMV bodies. Standalone post
accepts32/48 heads. Existing pool and constant offsets are unchanged. Explicit
byte adapters reconcile128 active state rows with140 production rows/head,
and gather separate bank-major Wa/Wb regions into standalone AB banks.

Tests first failed for missing geometry/state/AB adapters (3), missing token
program (1), missing signed-zero-aware post metric (1), and acceptance of
unsupported key-head/conv/epsilon tuples (3). All8 now pass. Adapters use
distinct random bytes across heads/banks and test untouched convolution and
padding. The program test checks all ten dispatches, distinct state input/output
BOs and absence of host math. Three cache-dependency tests cover precise LN.

## Fixture and numerical contract

`utilities/test-wide-deltanet-layer.py` uses seed38427, synthetic production
Q4 pools, bf16 AB/conv/norm constants, and independent FP64 arithmetic with
casts at physical f32/BF16 boundaries. It runs four cold and four warm tokens;
NPU-produced convolution/recurrent state persists between tokens. An extra
cold reset/repeat must reproduce output and padded state bit-for-bit. It hashes
kernels, cfg, weights, inputs and references, clears stale outputs, NaN-poisons
outputs and guards each with64 bytes. No reference file is loaded by the cfg.

Unchanged full-layer thresholds come from `designs/layer_x/compare.py`:
xn maxrel<8e-3; residual/xm/conv/state<2e-2; final output<5e-3; cosine>0.9999.
Strict conditional comparisons of QKV/Z/out/down and full FFN use the actual
device inputs *after inference*, maxrel<1e-4 and cosine>0.9999999. These isolate
primitive accuracy; they do not substitute for the independent layer reference.
Intermediate and per-head metrics remain diagnostic and are serialized.

## Observed precision failures and fixes

The first run completed90 NPU dispatches with intact guards, but warm token1
had output maxrel0.08514535. Its first divergence was entry RMSNorm BF16
rounding. Streamed wide LN now uses the existing three-component/six-product
helper for squares and weighted normalization, with cache dependencies updated.
All12 existing LN hardware inputs pass; xn matches numerically/bitwise in them
and all eight complete-layer entry inputs. This alone reduced the failing
output to0.00539170, still above0.005.

The standalone post worker now opts into the precise helper for squares,
products and SiLU (`POST_PRECISE=1`). The fused caller defaults to the original
arithmetic, so legacy fused models are not silently switched. Both standalone
32/48 probes pass random, zero-output and saturated ±20-gate cases: nine checks,
including exact prefix equality, with zero numerical BF16 mismatches against
the independent reference. Signed +0/-0 counts as equal; the historical8e-3,
cosine0.999999 and <5% mismatch limits remain unchanged.

However, **the final layer still fails**, now at warm-1 maxrel0.00908004
(0.908%, limit0.5%). More accurate local math does not guarantee monotonic
improvement after downstream BF16 rounding. Do not choose a fixture, arithmetic
variant or tolerance just to cross this boundary. Remaining differences start
at Q4 projections and propagate through conv/state and FFN; the precise cause
of the final amplification still requires isolation.

| Field, over eight tokens | Worst maxrel | Minimum cosine |
| --- | ---: | ---: |
| Entry xn | 0 | 1 |
| Mixer residual | 2.73541e-4 | 0.9999999527 |
| Post-FFN input xm | 6.17284e-3 | 0.9999995102 |
| Recurrent state | 1.66928e-3 | 0.9999999709 |
| Convolution state | 3.125e-3 | 0.9999999906 |
| Final output | **9.08004e-3 — FAIL** | 0.9999999632 |
| Conditional QKV | 5.30416e-6 | 0.99999999998 |
| Conditional Z | 5.82094e-6 | 0.99999999998 |
| Conditional output projection | 1.56161e-5 | 0.99999999984 |
| Conditional down | 3.60082e-5 | 0.99999999997 |
| Conditional full FFN | 3.46989e-5 | 0.99999999997 |

873 of874 checks pass; the failing check is `warm-1/y`. All canaries, state
copies, zero padding, convolution current-token copies and reset/repeat checks
pass. Stricter state head-local1e-4 diagnostics pass only3/8 token tensors;
this is explicitly retained, alongside the earlier [A7 caveat](wide-deltanet-a7.md).

## Physical resources and reproduction

Measured from addressed MLIR and core ELF (`llvm-size`), not source estimates:

| Standalone design | Placed data end, including stack | Core text |
| --- | ---: | ---: |
| QKV/Z K5120/N16384, eight cores | 59392 B/core | 4144 B |
| Output K6144/N5120, eight cores | 59392 B/core | 3568 B |
| Precise LN5120 | 57472 B | 3808 B |
| Precise post48 | 22784 B | 6032 B |
| Precise post32 regression | 22784 B | 5664 B |

These are separate contexts reused sequentially, not a simultaneous fused
placement. Existing AB/glue/step and precise segmented FFN binaries are reused;
their hashes are recorded in the fixture manifest. Host BO copies are not a
performance optimization, and timings from this harness are not model latency.

Exact build/run commands: [open-wide-deltanet-layer skill](../../../.opencode/skill/open-wide-deltanet-layer/SKILL.md).
Artifacts/traces/manifests/results live under
`open_kernels/designs/wide_deltanet/build_layer/`; original failing evidence is
in `baseline/`, LN-only evidence in `norm_only/`, current results in `acceptance/`.
Toolchain: Python3.14.4, mlir-aie1.4.2, llvm-aie21.0.0.2026080301+c9c5ecb7,
XRT2.26.0, Ryzen AI9 365/Strix, firmware1.1.2.64. Hardware runs used the host
open XRT harness; Docker was unnecessary. No new closed dependency was used.

Full open-engine regression: **754 passed,47 skipped**. This CPU result does
not turn the failing hardware acceptance into a pass. Keep full attention
layer, 8-layer slice, autoregressive/model validation, runtime integration and
model packing pending. The original fused `lx` shim DMA guard remains in place.

SHA256 of final evidence (generated files remain ignored):

```text
3e0103b4eaa1a08b18f4d4688fd2f48c9056c0552f2a8b1270c6da98dbeaa88d  acceptance/layer-fixture.json
d3fa3f209652edeb3c46d6989557d9ce3be0c5ff3a08785dfc3d83f10fa3b182  acceptance/layer-results.json
9837fe8fba22807eab0e4c22682d89eb31a9b8fb0251e2fe8a01a9dd97945c4c  acceptance/post-results.json
6fafbf8da8fd80834d6ed623b0f6cb7a8639f7ea68f4f9899c78e596ebd77e65  ln_precise/ln-results.json
```
