# Retained prefill diagnostics

For full-depth native experiments, start with the [combined CLI](README.md) or
[expanded CLI](../prefill-expanded-cli/README.md). The CPU diagnostics below
remain support tools for their independent references and isolated replays.
They do not establish current production acceptance.

## Why these tools remain

The combined/expanded CLIs exercise whole-model propagation. They do not replace
isolated two-slot projection replay, per-head rounding attribution, or capture-only
state observations. Their builders and preparation also import the earlier tools:

| Tool | Remaining dependency or independent check |
|---|---|
| `prefill-prefix-scan` | Validator/scorer imported by projection attribution and sequential drift; its source hash is pinned in retained attribution records. |
| `prefill-projection-attribution` | Produces per-head FP64 predictions and NPZ references consumed by entry/prefix replay. |
| `prefill-combined-attribution` | Produces down-projection reference NPZ; down replay imports its capture validator and packing oracle. |
| `prefill-entry-replay` | Combined CLI imports its low-weight builder; prefix replay uses its preparation helpers and isolated native controls. |
| `prefill-prefix-replay` | Produces the causal gated-prefix baseline consumed by output replay; output/down replay import its reference helpers. |
| `prefill-out-replay` | Combined CLI imports its builder and recursive historical-source verification. |
| `prefill-upgate-replay` | Supplies the generic harness builder to FFN replay/sweep and refined-sublayers, plus input/control helpers to paired weights. |
| `prefill-paired-weights` | Supplies low-weight builders to the combined CLI and narrow replays, and independent paired-input controls to FFN checks. |
| `prefill-combined-sublayers` | Supplies intermediate captures and the independent audit consumed by combined attribution and attention/down evidence. |
| `prefill-down-replay` | Expanded CLI imports its builder; attention replay consumes its checker and retained native results. |
| `prefill-attention-replay` | Expanded CLI imports its builder and prepares from its accepted replay and CPU audit. |
| `prefill-paired-input` | Supplies the paired-input implementation to residual tracing and the retained control used by expanded/slice boundary checks. |

Retained reports bind source paths and SHA256 values. The historical-source
verifier also requires the current tracked file to exist. Removing an apparently
old script can therefore break preparation even after its Python imports have
been replaced. Keep these sources and their oracle, packing, causal-state and
provenance tests until both executable consumers and evidence dependencies have
been migrated. Never rewrite saved source hashes to make a cleanup pass.

The three former attribution runbooks are consolidated below. Commands still
use the original source paths and create fresh output directories under `Models/`.
Measured checkpoints describe those historical captures, not today's model status.
Current production/block baselines, semantic run fixtures, `tail_crossover.py`
and `tail-crossover.json`, and the historical `prefill-perf/e2e-profile-v1.json`
remain intentional evidence.

## Shared execution rules

These CPU-only tools stream one layer's effective weights at a time. Use one
OpenBLAS thread, a fresh output directory and unchanged audited captures.
They revalidate native builds, compiler/kernel dependencies, primary references
and capture identity before attributing error. Failed runs persist incomplete
reports; successful exit means diagnostic completion, with `model_acceptance`
always false. Never rewrite historical source hashes or promote a conditioned
suffix to a model reference. NPZ files preserve signed vectors and predictions.
All original maxrel bounds remain <0.005.

## Causal prefix scan

The scan reuses the independent layer oracle on the retained weighted-FFN
candidate's native residuals, without building or dispatching native code.

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

The report records complete coverage, first failures and the top ten values
per component/route. Tests in `test_prefill_prefix_scan.py` retain causal carry,
reset, omitted positions, cancellation, zero-reference and failure coverage.

### Validated checkpoint (2026-10-21)

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

Evidence, vectors and logs are retained under
`Models/qwen38-27b/prefill-prefix-scan/`.

## Projection rounding

Analyze L34/P17, L37/P7 and L53/P17 using one representative of the identical
hybrid schedules. This tool does not measure sequential projections.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-projection-attribution/check.py \
  --run Models/qwen38-27b/prefill-prefix-sublayers/run \
  --build Models/qwen38-27b/prefill-prefix-sublayers/build \
  --baseline Models/qwen38-27b/prefill-weighted-cli/v2/candidate/run \
  --out Models/qwen38-27b/prefill-projection-attribution/run
```

### Projection prediction

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
maxrel bounds apply. The L2 remaining-error ratio compares prediction gap
to native-minus-exact error; undefined ratios are null, never NaN/infinity.

### Causal substitutions

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

`test_prefill_projection_attribution.py` protects paired-input/weight separation,
small-head gates, causal reset, signed interaction and failure persistence.

### Validated checkpoint (2026-10-21)

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

The subsequent entry replay used N16384/K5120/T256 at L34/P6 and L37/P0
(norm rows from the L34/P17 and L37/P7 files), with byte-exact concatenated
QKV/Z controls and two slot placements. L53/P17 motivated output replay.
The historical full residual result was **3974/4032 FAIL**.
Evidence is under
`Models/qwen38-27b/prefill-projection-attribution/`.

## Combined rounding

Analyze retained combined sublayers at L63/P1, L42/P17 and L52/P17.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-combined-attribution/check.py \
  --run Models/qwen38-27b/prefill-combined-sublayers/native-v1 \
  --audit Models/qwen38-27b/prefill-combined-sublayers/native-v1/sublayer-audit-v2.json \
  --out Models/qwen38-27b/prefill-combined-attribution/v1
```

Require unchanged audited sources, build dependencies, prepared model/reference
provenance, all baseline outputs and all 83 sublayer captures. Verify 40 hybrid
pairs before using hybrid as representative of both schedules.

Dequantize each manifest packing operation and require equality with the source
reader's effective weights. Attention Q and gate share a tensor: use `chunk0`
to select their distinct row ranges. Check all eight operations (three FFN down,
Q/K/V/gate and attention output), retaining packed/effective hashes.

Compare three FP64 predictions: FP32 input with effective weights, paired-BF16
input with effective weights, and paired-BF16 input with BF16-rounded weights.
Retain separate per-head reference scales. A close fit of the last prediction
to native output supports weight-rounding attribution; it is not byte-exact
replay of native accumulation. Report native error and prediction fit separately.

Reconstruct three selected independent CPU outputs bytewise. Preserve signed
vectors in NPZ files: propagated error, FFN front, paired input, weight rounding,
remaining native difference and residual addition. Their vector sum must close
against the full error; magnitudes do not add. For L63 also decompose the
pre-FFN residual across entry, host attention and output projection. Causal
substitutions use only positions 0..P, fresh paired KV and FP32 projection/og
storage boundaries. Evaluate one-role and joint ideal attention substitutions
followed by ideal FFN. These conditioned suffixes are diagnostics, never
replacement model references or evidence of fresh full-depth acceptance.

### Measured results (2026-10-23)

All **8/8 manifest/source identities**, **1124 projection/head comparisons**,
three independent CPU output reconstructions and **4 signed decompositions**
complete. Every local native projection passes the original 0.005 bound.
Full-depth acceptance remains **4000/4032 FAIL** from the unchanged native run.

| Role | Comparisons | Worst native maxrel | Worst BF16 prediction gap | Worst remaining L2 error fraction |
|---|---:|---:|---:|---:|
| Q | 480 | 0.0023476238 | 2.8980e-6 | 0.2925% |
| K | 80 | 0.0018736759 | 2.4349e-6 | 0.2321% |
| V | 80 | 0.0010985654 | 3.0213e-6 | 0.3126% |
| Gate | 480 | 0.0015700262 | 2.4409e-6 | 0.1506% |
| Attention out | 1 | 0.0006512213 | 1.7816e-6 | 0.0723% |
| FFN down | 3 | 0.0039805268 | 3.3603e-6 | 0.1205% |

The remaining L2 fraction is the norm of native-minus-rounded prediction divided
by native-minus-effective prediction, per head/position. It is not a percentage
of full-model error explained.

L63/P1 output error **0.0096485275** includes propagated/ideal-FFN error
**0.0070258073** and down-weight rounding component **0.0026236663**, measured
on the same independent output scale. The remaining down implementation
component is **2.2129e-6**. L42/P17 and L52/P17 weight-rounding components are
**0.0005789814 / 0.0002769049**; ideal FFN still leaves **0.0052107169 /
0.0123487787**. Fixing down alone therefore cannot close these retained errors.

At L63/P1, joint ideal Q/K/V/gate, attention output and FFN on the actual
incoming prefix gives **0.0040534510**. Replacing gate alone while using ideal
attention output and FFN gives **0.0049558575**; Q/K/V substitutions separately
remain **0.0068323553 / 0.0070201505 / 0.0059709254**. These experiments preserve
the accumulated incoming error and do not establish causal propagation after
correction across the full model.

The subsequent down replay used N5120/K17408/T256 on these three H rows,
with byte-exact controls, two placements, inactive-column checks and composed
residual gates. Attention replay covered fused entry N14336/K5120 and output
N5120/K6144 with causal prefixes before the expanded CLI integrated them.
`test_prefill_combined_attribution.py` protects packing, causal attention and
signed attribution invariants. Neither isolated acceptance nor this CPU result
establishes full-model acceptance.

Evidence: `Models/qwen38-27b/prefill-combined-attribution/v1/`.


## Block decode checks

Use one numerical checker for both post-prefill decode gates:

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/decode_transition.py \
  --check state --prompt 257 --decode 2 --layers 8
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/decode_transition.py \
  --check logits --prompt 700 --decode 16 --layers 64
```

`state` (the default) requires exact greedy-token agreement and compares final
conv/recurrent/KV state as well as all logits. `logits` uses the same unchanged
oracle cosine bound but reports argmax disagreement separately. It replaces
the separate 16-token logits entry point. Defaults remain 40/2/8 for
prompt/decode/layers in state mode and 700/16/64 in logits mode. The shipped
`oflm` text-output check remains separate because it cannot expose these tensors.

The checker forces block-major execution on the block side. Only `--chunk 256`
is supported: earlier arbitrary chunk values changed report metadata without
changing the native CLI's physical schedule. `--position` shifts the absolute
state position; prompt dump indices remain relative. Missing, empty, malformed
or incomplete dumps fail, and an existing output directory is never reused.

Result tool IDs and the separate state/logits acceptance rules remain compatible
with retained evidence. Historical JSON is not rewritten. Host tests exercise
both commands through a subprocess CLI double and real binary dump readers;
these tests are software checks, not NPU accuracy evidence.
