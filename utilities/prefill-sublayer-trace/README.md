# Selected causal prefill sublayers

This private CLI extends the per-layer residual tracer at zero-based targets
L63/P7, L60/P38 and L35/P80. It preserves the paired-input arithmetic and all
causal state evolution. Sequential NPU additionally dumps each selected layer's
complete activation scratch; both hybrid schedules dump input, pre-FFN residual,
post-attention norm, separate up/gate projections, activation and down output.
No weights, production source, kernel binaries or independent references change.

## Build

Use a fresh directory, GCC 15.2.0, CMake Release/OpenMP and the local XRT SDK:

```bash
ironvenv/bin/python utilities/prefill-sublayer-trace/build.py \
  --out Models/qwen38-27b/prefill-sublayer-trace/verified \
  --xrt-include /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  --xrt-lib /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib \
  --jobs 4
```

The parent builder and its source anchors are reused without editing them.
The new capture header is an explicit compiler dependency. Evidence pins the builder
and capture helpers, original/copied sources, CLI, compiler and CMake cache.
Existing output directories are rejected before touching their evidence.

The measured CLI SHA256 is
`dee5a361dcbf5bba6fc8b2e29e182b616652679c05186291d2b88a7091697d7d`.
The original `instrumented` build and the `verified` build are byte-identical.
The latter fixes builder-only protection against reusing an existing directory;
use its current evidence for validation. Native capture used the identical first
binary. Both builds pass CTest 4/4. Keep all measured directories intact.

## Run

Requires at least 35 GiB available RAM and exclusive, serial NPU processes.
Use a new output directory for each attempt; never overwrite the retained
per-layer baseline or substitute CPU states in the native engine.

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
ironvenv/bin/python utilities/prefill-sublayer-trace/run.py \
  --build Models/qwen38-27b/prefill-sublayer-trace/verified \
  --baseline Models/qwen38-27b/prefill-residual-trace/run-257 \
  --out Models/qwen38-27b/prefill-sublayer-trace/run-257
```

The runner copies independent references and uses the established three-route
prefill checker, including all 257 prompt logits and one NPU continuation.
It strips ambient OFLM_OPEN flags and selects NPU continuation explicitly.
Original numeric failures remain visible. A completed numeric FAIL from the
prefill checker still proceeds to the selected-sublayer checker.

## CPU audit

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sublayer-trace/check.py \
  --build Models/qwen38-27b/prefill-sublayer-trace/verified \
  --baseline Models/qwen38-27b/prefill-residual-trace/run-257 \
  --run Models/qwen38-27b/prefill-sublayer-trace/run-257 \
  --out Models/qwen38-27b/prefill-sublayer-trace/run-257/sublayer-audit.json
```

Require all **777 logit files** and **192 residual files** byte-exact against the
controlled baseline before interpreting new boundaries. Require exactly **39**
selected capture files, correct sizes, finite values, identical hybrid captures,
hybrid input identity with the previous layer and reconstruction of each layer
output from pre-FFN residual and down output within FP32 summation error bounds.
The scratch layout and FP32 norm are those of the verified production export.

For each selected layer, independently replay its complete CPU input prefix with
its own zero-initialized conv/S/KV history. Its final output must reproduce the
retained CPU reference bytewise. This separates global reference drift at input,
pre-FFN and output boundaries. A second replay uses the actual native input prefix
and independent ideal CPU history: `ideal_state_from_native_input` is a conditioned
diagnostic, **not captured native state or an alternative acceptance reference**.

FFN gates use FP64 weights and unrounded normalization from the actual captured
pre-FFN residual. Check norm, combined front activation, down projection from
captured activation, and composed FFN. Hybrid captures additionally check up,
gate and SiLU from captured up/gate. Sequential up/gate are not retained in the
production scratch; do not claim separate checks for them. All numeric gates
remain maxrel <0.005. A passing conditioned FFN does not establish global model
accuracy or prove attention is faulty. The checker always reports model
acceptance as false for this diagnostic scope.

CPU work holds one layer's dequantized weights at a time. It verifies source,
reference, compiler-dependency and capture hashes. Exceptions invalidate old
success. `capture_valid` and `conditioned_ffn_passed` are separate flags.

## Tests

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q \
  specs/open-engine/tests/test_prefill_sublayer_trace.py
```

Eight tests cover target selection, padding, nonfinite and duplicate output,
private patch drift, independent FFN gates, stale reports and safe build refusal.
Seven cases were observed RED before their implementation/fix. Full host suite:
1484 passed / 47 skipped. Native CTest: 4/4 in both builds.


## Measurements — 2026-10-20

All three native routes complete, with **777/777** logit files and **192/192**
residual files byte-exact to the previous controlled trace. All **39** selected
captures are complete; **18/18** hybrid sublayer pairs agree bytewise. Three
independent causal CPU reconstructions reproduce their stored outputs exactly.
All **54/54 conditioned FFN gates** pass over nine route/target cases; worst
normalized error is **0.0024681085**, hybrid front activation at L35/P80.

| Selected failure | Pre-FFN error | Native output error | Ideal FFN on native pre-FFN residual: output error | Native vs that ideal output |
|---|---:|---:|---:|---:|
| Sequential L60/P38 | 0.0093615690 | 0.0090391466 | 0.0090472423 | 0.0000081696 |
| Sequential L35/P80 | 0.0084320701 | 0.0051683727 | 0.0051656855 | 0.0000027012 |
| Hybrid L63/P7 (both schedules) | 0.0008105224 | 0.0058400461 | 0.0046310385 | 0.0031273746 |

These output diagnostics use the existing oracle's FP32 norm/activation
boundaries and FP64 arithmetic; conditioned FFN gates separately use unrounded
FP64 normalization. Each metric normalizes by its own reference maximum, so
these scalars are not additive error contributions.

Sequential failure is already present before FFN, and the ideal FFN reproduces
its output error closely. For hybrid L63/P7 the ideal-FFN counterfactual passes
the global residual gate while the native output fails: local FFN arithmetic
contributes to the threshold crossing despite passing its conditioned gates.
This does not prove a uniquely faulty operation. Next separate up/gate,
activation and down contributions at that hybrid target before choosing a
precision change; sequential upstream causal-state/input drift remains open.

The run's initial automatic audit refused the old builder-source hash after the
existing-directory protection was fixed. The refusal is retained in
`initial-build-provenance-refusal.json`. Rebuilding produced the **same CLI hash**;
a separate CPU audit with the current `verified` build passes capture validity
and conditioned FFN checks. Native captures were not edited or rerun under a
different binary. Full-model accuracy remains FAIL (558/774 logits on this
fixture), regardless of the successful diagnostic checks.

Evidence is under `Models/qwen38-27b/prefill-sublayer-trace/`:
`run-257/sublayer-audit.json`, `run-257/report.json`, both build records, source/
compiler-dependency hashes, RED logs, host/CTest logs and `validation-evidence.json`.

## CPU vector attribution

Reuse these captures without a resident model or a new native build:

```bash
mkdir -p Models/qwen38-27b/prefill-ffn-attribution
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sublayer-trace/attribute.py \
  --run Models/qwen38-27b/prefill-sublayer-trace/run-257 \
  --build Models/qwen38-27b/prefill-sublayer-trace/verified \
  --baseline Models/qwen38-27b/prefill-residual-trace/run-257 \
  --out Models/qwen38-27b/prefill-ffn-attribution/attribution.json
```

The analyzer verifies the retained source/build/reference/capture hash chain,
baseline logit/residual identity and independent causal reconstruction record.
It streams one layer's FFN weights at a time and analyzes the three hybrid rows.
Both hybrid schedules share the result only after their captures and layer
outputs are proved identical. No CPU history is inserted into native execution.

Let `un, gn` be ideal projections of captured norm, `uc, gc` the captured
projections, `du = uc-un`, and `ds = SiLU(gc)-SiLU(gn)`. Activation discrepancy
from projections is exactly:

```text
du * SiLU(gn) + un * ds + du * ds
    up-only    gate-only  interaction
```

Project each term through the same FP64 down matrix. Also retain norm error,
captured activation minus `uc*SiLU(gc)`, captured down minus ideal down of
captured activation, and final residual-add rounding. The baseline is the
existing oracle FFN on captured pre-FFN residual, preserving its FP32 norm and
activation boundaries; a separate bridge accounts for the oracle activation
rounding. This bridge is bookkeeping, not a native defect.

The complete vectors, including propagated baseline error, must sum to native
output minus the stored independent **FP32** output within an FP64 reduction
bound. This stored reference can differ slightly from the preceding checker's
unrounded causal output. Every component uses one common final-output reference
scale. Maxima are still non-additive; signed components at the worst native
coordinate and all vectors (`attribution.npz`) preserve cancellation information.
The `passed` fields on component magnitudes only apply the same 0.005 threshold;
they are not new primitive acceptance gates.

Two diagnostic counterfactuals remain distinct: ideal suffixes recompute all
downstream operations on CPU; algebraic component removals freeze observed
downstream errors. A real NPU replay can change those errors, so neither predicts
the result of a kernel correction. Removing up or gate includes its interaction.
Attribution validity never promotes model acceptance. Exceptions invalidate old
success in the output JSON.

### Attribution results — 2026-10-20

All three vector decompositions close; largest absolute closure discrepancy is
**3.98e-14**. At L63/P7 the retained output scale is **44.3703270** and the worst
native coordinate is **3994**. The two schedules are byte-identical.

| Component | Maximum absolute contribution / common scale | Signed contribution at coordinate 3994 / common scale |
|---|---:|---:|
| Propagated ideal-FFN baseline | 0.0046311444 | -0.0027144259 |
| Up projection | 0.0044931327 | -0.0044931327 |
| Gate projection | 0.0002682804 | +0.0000108906 |
| Up/gate interaction | 0.0000014409 | -0.0000014409 |
| Activation | 0.0000000619 | -0.0000000619 |
| Down projection | 0.0013581369 | +0.0013581369 |

Norm, oracle-rounding bridge and residual-add terms are each below 5.1e-8
on this scale. Up is the largest local contribution; down partially cancels it.
Native output maxrel is **0.0058400461**. Ideal down on captured activation
increases it to **0.0071981830**. Ideal up → activation → down with captured gate
reduces it to **0.0046451266**; ideal gate → activation → down with captured up
gives **0.0072075708**. Algebraically removing up including interaction while
freezing downstream errors gives **0.0045399893**.

The L60/P38 and L35/P80 hybrid control outputs remain below threshold
(**0.0014798680 / 0.0001922195**). No correction has been executed on the NPU;
global logits remain **558/774 PASS**, so full-model accuracy still fails.
Next replay the L63/P7 up/gate GEMM with production layout, captured norm and a
byte-exact control before changing precision. Preserve T256 and row position 7;
if filling other rows with zeros, first prove the selected result is unchanged.
This attribution does not distinguish weight packing from GEMM arithmetic.
Sequential propagated-state/input failures require a separate investigation.

Tests: **16 new cases**, 15 observed RED before implementation; full host suite
**1500 passed / 47 skipped**. No new native build or NPU dispatch. Reports,
complete contribution vectors, provenance and test logs are retained under
`Models/qwen38-27b/prefill-ffn-attribution/`.
