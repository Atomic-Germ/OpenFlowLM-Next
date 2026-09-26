# Wide DeltaNet layer precision — 2026-09-26

**The synthetic B7 one-DeltaNet-layer gate passes: 874/874 checks.** This
closes the numerical failure in [the initial layer report](wide-deltanet-layer.md).
It uses the same seed38427, eight cold/warm tokens, production Q4 pools, FP64
oracle, physical f32/BF16 boundaries, and inherited thresholds. The full
attention layer, 8-layer slice, model/runtime integration and packing remain
pending. No model or catalogue entry is promoted.

## Diagnosis and changes

`utilities/diagnose-wide-deltanet-layer.py` verifies fixture/kernel hashes and
guard bytes, then replays captured boundaries offline. These conditional and
counterfactual results are diagnostic only; acceptance always uses the original
independent reference. The harness still performs only NPU dispatch and BO copies.

At baseline warm-1, replaying device QKV through an otherwise ideal suffix
produced maxrel0.01119558, and device convolution history produced0.01597707.
The Q4 activation table discarded small BF16 components when converting a
32-element block to int16. An opt-in second int16 table retains the residual
at a scale15 bits finer. A third BF16 block-sum/product component and compensated
accumulation reduce the remaining projection error. Four32-row compensation
slots persist across K tiles and reset for each output band. The table grows
from18432 to22592 bytes for the H5120 projection; pool and host BO formats stay
unchanged. `PROBE_Q4_CORRECTION=1` is restricted to the standalone dense Q4
H5120/K5120 probe, with an explicit L1 budget guard.

Improving only this projection did **not** close the gate. Recorded experiments:

| Variant | warm-1 final maxrel | Result |
| --- | ---: | --- |
| Original precise LN/post, normal Q4/glue/step | 0.0090800402 | FAIL |
| Q4 residual and third component | 0.0121560168 | FAIL |
| Q4 plus compensated sum | 0.0120876014 | FAIL |
| Above plus precise glue and recurrence | **0.0001712260** | **PASS** |

After Q4 compensation, warm-0/1 convolution BF16 casts match the reference.
Warm-1 replay with only device QKV or convolution history becomes exact. The
remaining error comes from conv/record arithmetic and accumulated recurrence:
device previous recurrent state alone gives maxrel0.00911697.

`DNGLUE_PRECISE=1` retains three components in the f32-by-BF16 convolution,
uses the existing precise SiLU helper, and improves Q/K square products and
normalization. `DN_STEP_PRECISE=1` uses three-component/six-product arithmetic,
an FP32 delta and compensated reductions for S^T k and S'^T q. Its existing
three256-BF16 scratch arrays hold FP32 delta and compensation values, preserving
the standalone ABI and placement. Both modes default off. Their build keys
include the flag and precise helper dependencies. Legacy fused callers retain
their arithmetic, and the fused wide `lx` DMA guard remains in place.

## TDD and validation

The existing full-layer test supplied the failing hardware case throughout.
The new table budget test first failed for the missing sizing helper, then
again when compensation required another512 bytes; it now verifies default
sizing, corrected sizing and rejection of an oversized table. No seed, reference,
threshold or acceptance check was relaxed.

Final measurements over eight tokens (plus the cold reset/repeat):

| Field | Worst maxrel | Minimum cosine |
| --- | ---: | ---: |
| Entry xn | 0 | 1 |
| Mixer residual | 1.35176e-5 | 0.999999999906 |
| Post-FFN input xm | 3.08642e-3 | 0.999999979149 |
| Recurrent state | 1.08385e-5 | 0.999999999999 |
| Convolution state | 4.88281e-5 | 0.999999999999 |
| Final output | **1.71226e-4** | **0.999999999952** |
| Conditional QKV | 2.10545e-7 | 0.999999999999993 |
| Conditional Z | 1.82378e-7 | 0.999999999999993 |
| Conditional output projection | 1.58788e-5 | 0.999999999855 |
| Conditional down | 3.62449e-5 | 0.999999999972 |
| Conditional full FFN | 3.76975e-5 | 0.999999999971 |

All canaries, state adapters, padding and reset/repeat checks pass. The stricter
head-local state diagnostics also pass for all eight tokens. Conditional warm-1
glue/step/post diagnosis and the intermediate metrics remain in the JSON evidence.

Additional hardware regression: corrected K5120/N16384 projection9/9 inputs
(worst maxrel1.74355e-7), default K5120/N1024 projection9/9, and rebuilt default
32-head recurrence4 persistent tokens/8 state-output checks. Open-engine CPU
suite: **760 passed,47 skipped**. These are synthetic checks, not model inference
or a performance claim.

## Resources and reproducibility

Addressed MLIR and core ELF measurements, including allocated stack:

| Design | Placed data end | Core text |
| --- | ---: | ---: |
| Corrected QKV/Z K5120/N16384 | 63552 B/core | 4368 B |
| Precise48-head glue | 59776 B | 8232 B |
| Precise48-head recurrence | 43776 B/core | 3680 B |

Toolchain remains Python3.14.4, mlir-aie1.4.2,
llvm-aie21.0.0.2026080301+c9c5ecb7, XRT2.26.0; Ryzen AI9 365/Strix,
firmware1.1.2.64. Host open XRT harness, no closed kernel dependency.
Exact commands and operational constraints are in
[open-wide-deltanet-precision](../../../.opencode/skill/open-wide-deltanet-precision/SKILL.md).

Ignored evidence directories under `open_kernels/designs/wide_deltanet/`:
`build_layer/acceptance` (baseline), `build_layer_corrected/acceptance`
(residual-only failed experiment), `build_layer_compensated/acceptance`
(compensated Q4 with legacy glue/step), `build_layer_precise/acceptance`
(passing composition), and `build_default_regression/` (legacy checks).
Keep failed experiments when investigating rounding; do not select an arithmetic
variant just because it happens to cross a tolerance.

SHA256 under `build_layer_precise/acceptance`:

```text
ffe8f4554c8b6e67955a84c2ef6c1e55f8ae1406943b04036307247867543cf3  layer-fixture.json
fc18044efd0d69b3728c74e01806bafcb85e025be5ca5734a8c1ba312ee2cc08  layer-results.json
19d72dd874de9dca0e695eade69758eeec0bce9bdfcf18139dc665a08b8b3b2c  warm-1-diagnosis.json
```
