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
