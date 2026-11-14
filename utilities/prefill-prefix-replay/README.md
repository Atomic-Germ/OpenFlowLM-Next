# Causal QKV/Z prefix replay

Replay the three complete captured 20-token prefixes at L34/P17, L37/P7 and
L53/P17. Apply paired-weight QKV/Z to every real token, run the actual production
host DeltaNet, then the original NPU output projection. Incoming residuals and
entry norms remain the retained native inputs. This is a conditioned sublayer
experiment, not corrected cross-layer propagation or full-model acceptance.

## Reuse verified builds

No new binaries are built. Reuse the entry experiment's N16384/K5120/T256
control/low kernels, production N5120/K6144/T256 output kernel, standard harness
and `prefill-host-boundaries/wide/libprefill_boundaries.so`. Verify source,
compiler dependency, generated-source and binary hashes before dispatch. Reject
stale builds and existing result directories. Model/capture/reference provenance
is revalidated; packed effective QKV/Z/out weights must match source and the
retained CPU attribution. Only one layer's weights are unpacked at a time.

The host bridge uses production `block_host.cpp` with FP32 conv history and
`-ffp-contract=off`. It is loaded only after its recorded sources and binary
match. Require its **entire 20-row native gated output, conv history and raw
recurrent state** to match captures bytewise before NPU precision interpretation.
Captured final state belongs to P19, not the selected residual row.

Derive alpha/beta `lanes` from the transposed real tensor shape, as native core
does: **48** for this model, not the synthetic fixture/NPU recipe's padded 64.
Check every constant's extent and finite values before crossing the C ABI.
The initial attempt (`run/`) was rejected by exact native-output control with
zero NPU dispatches; its evidence is retained. A regression test now rejects
wrong constant strides before any native call.

## Run

Use exclusive NPU access. The largest single dispatch uses about 68.5 MiB
logical BOs (excluding runtime overhead); this does not require full-model
residency. CPU effective weights/references require additional memory.

```sh
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE ironvenv/bin/python \
  utilities/prefill-prefix-replay/check.py \
  --out Models/qwen38-27b/prefill-prefix-replay/v2 \
  --attribution Models/qwen38-27b/prefill-projection-attribution/run \
  --capture Models/qwen38-27b/prefill-prefix-sublayers/run \
  --baseline Models/qwen38-27b/prefill-weighted-cli/v2/candidate/run \
  --capture-build Models/qwen38-27b/prefill-prefix-sublayers/build \
  --build-root Models/qwen38-27b/prefill-entry-replay/builds \
  --harness-build Models/qwen38-27b/prefill-upgate-replay/harness \
  --bridge Models/qwen38-27b/prefill-host-boundaries/wide
```

There are **60 serial dispatches**: per layer, two placements × six entry
products and two placements × four output products. Place all 20 rows at
columns 0–19, then 128–147. Every product must be finite and zero outside real
columns. Fresh production entry sums must reproduce all captured QKV/Z rows;
new-control individual products must match production. Combine candidate in
FP32 as `(WhXh+WhXl)+(WlXh+WlXl)`; placement results must match bytewise.

Primary entry gates use retained original-FP32-input/effective-weight FP64
references, with separate heads and positions: **15360 checks** (7680 distinct
heads × two placements), maxrel <0.005 and rel_fro <=0.005. Strict improvement
is recorded separately. Keep this separate from conditioned host references.

Run control and candidate host blocks from independent zero state. Require
20-row block versus sequential host calls byte-exact in gated output, conv and
state; poison unused inputs with NaN and require zero output/state padding.
Compare each real head output and each final state head against an independent
FP64 reference driven by that variant's FP32 projections (maxrel <1e-3).
These **6048 conditioned host gates** do not replace the independent model
residual reference. Comparisons to ideal-entry host results are diagnostic.

Run original paired-input output GEMM on both full gated prefixes. Require
control selected output/residual byte-exact to native captures; placement
identity for all outputs. Check all **240 conditioned projection rows** and
selected residuals at unchanged 0.005 bounds. Selected residual references are
retained independent full-model FP64 pre-FFN values, not conditioned substitutes.

`complete`/`control_valid` indicate measurement validity, not numerical success.
Exit 1 preserves a complete valid measurement when a precision gate fails.
`model_acceptance` is always false. Save all inputs, outputs, configurations,
logs, state/reference NPZs and hashes. No corrected FFN/logits, full-model
resident run, production promotion or performance claim is included.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q \
  specs/open-engine/tests/test_prefill_prefix_replay.py
```

## Validated checkpoint — 2026-10-22

`v2/report.json` is complete with valid controls and all 60 dispatches recorded.
All **15360/15360 entry head gates** pass and strictly improve maxrel/rel_fro;
worst candidate values are **1.4844200e-5 / 9.4852176e-6**. All **6048/6048
conditioned host gates** pass (worst **6.0081719e-6**), including block-end state
at P19. Block/sequential host results and both NPU placements match bytewise.
All **240/240 output projection gates** pass, worst maxrel **0.0022730411**.

| Selected pre-FFN residual | Native control maxrel | Candidate maxrel | Candidate gate |
|---|---:|---:|---|
| L34/P17 | 0.0007177295 | 0.0006256207 | PASS |
| L37/P7 | 0.0010862168 | 0.0008810181 | PASS |
| L53/P17 | 0.0075342533 | 0.0074347548 | FAIL |

Each row was reproduced at both placements. L53 remains above 0.005; a complete
measurement therefore returns exit 1. Retained full-model residual acceptance
remains **3974/4032 FAIL**, not a new full-model run. The corrected candidate's
conditioned state error against ideal-entry host state is only 2.08e-6 to
4.02e-6, but that does not remove inherited input error or output weight rounding.
Next evaluate paired-weight output projection at N5120/K6144 with exact native
controls and these retained gated prefixes; preserve the independent residual
gate even if the local output projection improves.

Fourteen new RED-to-GREEN tests; full suite **1618 passed / 47 skipped**.
The first rejected geometry run and its extra failing regression test are kept
under the same artifact root. No binaries were rebuilt and no production model
arithmetic was changed. Result evidence pins 135 inputs/configs/logs and 168
artifacts, in addition to the validated capture/build dependency chains.
