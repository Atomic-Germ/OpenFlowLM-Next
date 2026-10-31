# Current L3 projection control

This private one-core replay runs the current Qwen3.8-27B Q/gate/K/V
projections from captured FP32 entry norms over positions 0–239. It uses the
current production-generated `dense_prep_f32` and `gemv_q4_gy` translation units,
unchanged Q4 headers, direct FP32 preparation and consistent quantized sums.
There is no candidate precision change or production integration here.

The four packed weight regions are extracted from the audited native L3 pool
and compared bytewise with independently packed source tensors. The replay
concatenates Q, gate, K and V bands on one core, retaining the production
64-row band law, 40 single-chunk calls per band and five FP32 input elements.
Byte-exact native outputs are required before interpreting this different
dispatch schedule. Every output and table buffer is poisoned before dispatch.

## Reproduction

Run from the repository root, using fresh output directories:

```bash
source utilities/env.sh
export OPENBLAS_NUM_THREADS=1

ironvenv/bin/python utilities/l3-projection-replay/build.py \
  --spec Models/qwen38-27b/production-post-fp32/kernels-v1/spec.json \
  --out Models/qwen38-27b/post-prefill-drift/l3-projection-build-v1

ironvenv/bin/python utilities/attention-entry-replay/resources.py \
  --build Models/qwen38-27b/post-prefill-drift/l3-projection-build-v1 \
  --out Models/qwen38-27b/post-prefill-drift/l3-projection-resources.json

ironvenv/bin/python utilities/l3-projection-replay/replay.py prepare \
  --capture Models/qwen38-27b/post-prefill-drift/native-l3-v1 \
  --build Models/qwen38-27b/post-prefill-drift/l3-projection-build-v1 \
  --resources Models/qwen38-27b/post-prefill-drift/l3-projection-resources.json \
  --out Models/qwen38-27b/post-prefill-drift/l3-projection-v2

ironvenv/bin/python utilities/l3-projection-replay/replay.py run \
  --out Models/qwen38-27b/post-prefill-drift/l3-projection-v2 \
  --harness Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1/run_kernel

ironvenv/bin/python utilities/l3-projection-replay/replay.py audit \
  --out Models/qwen38-27b/post-prefill-drift/l3-projection-v2
```

The input native replay must already have a complete independent audit and
240 exact full-layer residual controls. The harness must have valid compiler
dependency evidence. Preparation binds source files, model/export identity,
compiler dependencies, resource evidence, packed weights, norms, reference
projections and the exact command program. Do not edit pinned sources during
preparation, execution or audit. Compilation and hardware access require the
host's normal tool permissions. No full-model residency is required.

## Verified checkpoint — 2026-10-28

- Fresh private build: one core, 3376 bytes of program text, maximum L1 buffer
  end 32512, 6144-byte stack; resource audit passes. Build evidence binds 807
  input/dependency files and the binaries, linked ELF and address assignment.
- 240 NPU dispatches; all **960/960** individual Q/gate/K/V outputs are
  byte-exact to the current native L3 captures.
- All **240/240** activation tables exactly match the independent CPU
  construction: INT16 activations, shifts and both BF16 sum planes.
- Separate audit verifies all 480 output artifacts and repeats both comparisons.
- Logical NPU buffers total 45,964,544 bytes; this is not a host RSS measurement.
- TDD: 10 initial failing tests; 15 new tests pass. Focused suite: 49 passed.
  Full host suite: 1988 passed, 47 skipped.

`report.json` and `audit.json` retain `model_acceptance: false`.
`control_valid` measures reproduction of current arithmetic;
`table_valid` measures table identity. Neither demonstrates that the
projections are sufficiently accurate for full sequential-model propagation.
The global numerical failure is unchanged, and these instrumented timings
must not be used as a production performance benchmark.

Next, compare independent FP64 projections using the captured FP32 norm and
the exact reconstructed INT16 activation. Keep their signed difference
separate from the remaining native accumulation/epilogue error. P125/P205 are
the projection-attribution targets; P239 remains an attention/cancellation
regression control. Test a precision variant only after this attribution.

## Signed projection attribution — 2026-10-28

`attribute.py` reads the actual audited INT16/shift table, reconstructs its
activation in FP64, and independently decodes the packed Q4 weights. It
cross-checks every decoded weight against the existing Q4 decoder before
computing `W @ x` and `W @ x_quantized` over all 240 positions.

The signed decomposition is:

```text
activation_quantization = W @ x_quantized - W @ x
arithmetic_remainder    = native - W @ x_quantized
total                  = native - W @ x
```

All effects use the same reference scale within a role/position. The remainder
includes table sum-plane rounding, dot-product representation, epilogue and
accumulation; it is not an isolated FP32 accumulation measurement. The frozen
`without_quantization` counterfactual is not a corrected NPU execution.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/l3-projection-replay/attribute.py run \
  --capture Models/qwen38-27b/post-prefill-drift/l3-projection-v2 \
  --out Models/qwen38-27b/post-prefill-drift/l3-projection-attribution-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/l3-projection-replay/attribute.py audit \
  --out Models/qwen38-27b/post-prefill-drift/l3-projection-attribution-v1
```

The independent audit repeats all 960 cases and compares four NPZ array sets
byte-for-byte, including signed vectors and reference projections. INT16
activation quantization has greater maximum absolute error than the arithmetic
remainder in **960/960** cases; the ratio ranges from **2.97 to 19.24**.
At P125/P205, K quantization maxrel is **2.20503e-4 / 2.16412e-4** against
remainder **1.46467e-5 / 1.89299e-5**. These are conditioned projection errors,
not final residual errors or full-model acceptance measurements. P239 still
requires a separate attention/cancellation regression check.

Validation: 11 new tests fail before implementation and pass afterward; full
host suite **1999 passed, 47 skipped**. The next experiment is an isolated
two-pass INT16 residual-refinement projection with byte-exact first-pass
controls, followed by causal attention checks before any production change.

## Two-pass residual refinement — 2026-10-28

`refine.py` reuses the audited one-core binary. For each of 240 positions it
runs the original FP32 norm, then the exactly representable FP32 residual
`x - reconstructed_INT16(x)`. The candidate is one explicit **host FP32 add**
of the two NPU outputs. Every output/table buffer is poisoned before each
dispatch. The fresh first pass must reproduce all original projections and
tables bytewise; the second table must exactly match CPU preparation.

```bash
source utilities/env.sh
export OPENBLAS_NUM_THREADS=1
ironvenv/bin/python utilities/l3-projection-replay/refine.py prepare \
  --source Models/qwen38-27b/post-prefill-drift/l3-projection-v2 \
  --out Models/qwen38-27b/post-prefill-drift/l3-refinement-v1
ironvenv/bin/python utilities/l3-projection-replay/refine.py run \
  --out Models/qwen38-27b/post-prefill-drift/l3-refinement-v1 \
  --harness Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1/run_kernel
ironvenv/bin/python utilities/l3-projection-replay/refine.py audit \
  --out Models/qwen38-27b/post-prefill-drift/l3-refinement-v1
```

The numerical check independently decodes weights and compares both candidate
and control with FP64 projections of (a) the captured norm and (b) an unrounded
norm recomputed from the native layer input. It also checks the correction
projection itself. Four CPU FP64 attention trajectories start at position zero:
control, candidate, captured-norm reference and unrounded-norm reference.
Each uses its own full causal K/V history. Projection maxrel <0.005 and head
maxabs <0.02 / cosine >0.9999 remain unchanged.

Verified results:

- 480 NPU dispatches; **960/960** exact first-pass projection controls and
  **480/480** exact tables. The 240 residual inputs reconstruct the original
  FP32 norms exactly with their first-pass quantized activations.
- **960/960** candidate projections improve against both references. The
  captured-norm error reduction is **2.98–19.30×**; the unrounded-norm reduction
  is **1.95–18.89×**. All correction and primary projection gates pass.
- Both candidate CPU attention comparisons pass **5760/5760** heads. Maximum
  head error versus the captured-norm reference falls from **2.80667e-4** to
  **2.28459e-5**; versus the unrounded-norm reference from **2.79433e-4** to
  **5.31917e-5**. P125/P205/P239 each improve in these CPU attention comparisons.
- Separate audit repeats the references and measurements and compares all
  28 arrays in `numerics.npz` bytewise; all 961 output artifacts are bound.
- TDD: 11 initial failing tests; 14 final new tests pass. Focused tests:
  **40 passed**. Full host suite: **2013 passed, 47 skipped**.

`candidate_passed` covers this isolated experiment only. Native attention,
paired-KV storage, full-layer residuals, full-model propagation and production
performance have not been validated for the candidate. It doubles projection
dispatches and weight streaming in this harness; no speed claim is made.
No new kernels were built and production arithmetic is unchanged.

Next: replay corrected projections through actual NPU attention with causal
paired-BF16 KV and a byte-exact native control. Keep P239 as a distinct
attention/cancellation regression case before private full-ax integration.
