# Causal prefix drift scan

CPU-only diagnostics for the retained private paired-weight FFN candidate.
Reuses effective Q4NX weights and the existing independent layer oracle; it
neither builds native artifacts nor dispatches the NPU. Weights are decoded
and cached one layer at a time. Set OpenBLAS to one thread to reproduce the
retained reference arithmetic.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-prefix-scan/check.py \
  --run Models/qwen38-27b/prefill-weighted-sublayers/run \
  --baseline Models/qwen38-27b/prefill-weighted-cli/v2/candidate/run \
  --build Models/qwen38-27b/prefill-weighted-sublayers/build \
  --out Models/qwen38-27b/prefill-prefix-scan/run
```

Use a fresh output directory. Defaults cover layers 0–55 and positions 7/17
of the retained 20-token prompt. `--layers` and `--positions` can select a
smaller diagnostic scope. The checker verifies the sublayer audit, builds,
compiler dependencies, low-weight kernels, model/reference sources, native
logits and all 192 residual hashes. It rechecks exact hybrid/block-major
residual identity before using one hybrid representative.

For every layer, replay all preceding positions, starting with fresh independent
conv/recurrent/KV state for each route. Inputs come from the preceding layer's
retained residuals; layer zero uses model embeddings. Skip FFN only at positions
not selected for output comparison: its output does not update this layer's
attention state, and the next layer's inputs are supplied independently.
Reconstruct every selected independent CPU reference bytewise before comparing.

Each NPZ saves native, ideal-on-native-history and independent-reference outputs,
and three **signed vectors on the same output reference scale**:

- `propagated = ideal_on_native_history - reference`
- `ideal_history_gap = native - ideal_on_native_history`
- `total = native - reference`

The vectors must close within FP64 rounding error. Their scalar maxima need not
add: different coordinates can dominate, and errors can cancel. Input maxrel
has a different reference scale and is reported separately. The ideal-history
gap includes differences in local arithmetic **and** in state accumulated over
the prefix; it is not a conditioned single-operation error or actual native
state. Neither ranking nor threshold crossing identifies a unique faulty op.

The report records complete case coverage, first output failures by position,
and top ten values of each component per route. Exit zero means the diagnostic
and evidence checks completed. It does **not** mean numerical/model acceptance;
`model_acceptance` is always false. Interrupted/failed scans retain an incomplete
report and any already written vectors. Never substitute these diagnostic
outputs for primary references or infer long-prefix acceptance from this scan.

## Tests

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q \
  specs/open-engine/tests/test_prefill_prefix_scan.py
```

Tests cover causal conv/recurrent/KV carry and reset, omitted output positions,
invalid prefix coverage, common-scale cancellation, zero/non-finite references,
complete case inventories and persisted failures without overwriting evidence.

## Validated checkpoint (2026-10-21)

All 56 layers × two positions reconstruct **112/112 CPU references bytewise**.
The **224 cases** and signed-vector closure checks complete. Sequential passes
112/112 selected output gates (worst 0.0032302929 at L53/P17); hybrid passes
101/112, with first failures L40/P17 and L55/P7. All 64 hybrid/block-major
residual pairs are exact, so the hybrid diagnostic applies to both schedules.

| Hybrid row | Propagated maxrel | Ideal-history gap maxrel | Total maxrel |
|---|---:|---:|---:|
| L40/P17 | 0.0055866730 | 0.0003720786 | 0.0059587517 |
| L51/P17 | 0.0134731971 | 0.0004004110 | 0.0138594086 |
| L55/P7 | 0.0054887081 | 0.0007874112 | 0.0062761194 |

The largest hybrid ideal-history gap is L53/P17 (0.0009166706); the same row
has the largest sequential gap (0.0010513228). Hybrid P17 errors grow through
L33–40, while P7 shows earlier growth around L36–37. These nominate causal
sublayer captures L34/P17, L37/P7 and L53/P17; they do not prove a faulty kernel.
No candidate arithmetic or native artifacts changed. The full retained run's
residual acceptance remains 3974/4032, FAIL. Long-prefix acceptance remains open.

Eleven initial tests fail before implementation; thirteen final targeted tests
pass, including two added causal/zero-reference checks. Full host suite:
**1571 passed, 47 skipped**. Evidence, vectors and logs are retained under
`Models/qwen38-27b/prefill-prefix-scan/`.
