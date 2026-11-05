# Isolated paired-weight FFN down replay

Build private N5120/K17408/T256 control and low-weight kernels from copied
production design/dequant sources. Control uses BF16_RNE(W); low uses
BF16_RNE(W-BF16_RNE(W)) from the FP32 dequant accumulator. Keep the production
layout, Q4_1 format, B reuse disabled and unchanged inputs/weights/reference.
Record original/generated/compiler dependencies, binary hashes and all 32 core
resource budgets. Refuse existing build or result directories.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-down-replay/build.py \
 --mode control --out Models/qwen38-27b/prefill-down-replay/builds/control
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-down-replay/build.py \
 --mode low --out Models/qwen38-27b/prefill-down-replay/builds/low
```

Prepare captured H rows from L63/P1, L42/P17 and L52/P17. Revalidate the combined
sublayer and CPU-attribution provenance, model/manifest, packed/source weights
and primary references. Use slots P and P+128 in otherwise zero T256 inputs.
No full model residency is needed; only one projection weight/input/output set
is loaded per process (about 67 MiB of logical BOs, excluding XRT overhead).

Run serially under the shared device lock, checking exclusive NPU access.
Before interpreting candidates, require the fresh production and private control
high/low products to match individually. Their FP32 sum must equal the saved
native down output, and residual addition must reconstruct the saved native
layer output. Require zero inactive columns and both slot placements byte-exact.
Candidate is FP32((Wh*Xh+Wh*Xl)+(Wl*Xh+Wl*Xl)).

Keep original-FP32-input/exact-weight projection gates and independent composed
residual gates separate. Both maxrel and rel_fro must strictly improve for the
projection. A completed experiment may still fail residual accuracy; never
weaken bounds or claim model acceptance. Full-model propagation and long-prefix
acceptance remain separate. No production policy, kernel or model edit.

## Prepare, replay and re-audit

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-down-replay/check.py prepare \
 --attribution Models/qwen38-27b/prefill-combined-attribution/v1 \
 --capture Models/qwen38-27b/prefill-combined-sublayers/native-v1 \
 --audit Models/qwen38-27b/prefill-combined-sublayers/native-v1/sublayer-audit-v2.json \
 --out Models/qwen38-27b/prefill-down-replay/prepared-v2
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
 ironvenv/bin/python utilities/prefill-down-replay/check.py run \
 --prepared Models/qwen38-27b/prefill-down-replay/prepared-v2 \
 --build-root Models/qwen38-27b/prefill-down-replay/builds \
 --harness-build Models/qwen38-27b/prefill-out-replay/harness \
 --out Models/qwen38-27b/prefill-down-replay/native-v2
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-down-replay/check.py audit \
 --out Models/qwen38-27b/prefill-down-replay/native-v2
```

The reused harness must pass source/dependency/cache/executable hashes before
dispatch. Both run and audit return 1 when composed residual gates fail, even
with complete control/projection acceptance. Read `complete`, `control_valid`,
`projection_passed`, `improved` and `residual_passed` separately.

The first preparation wrote a FP32 oracle output into `output-reference.f64`.
All 36 initial dispatches and exact controls completed, but the checker correctly
refused its incomplete FP64 extent. Keep `prepared-v1`, `native-v1`, the failed
report and `reference-fix-snapshot.json`. The reference writer now explicitly
converts `.f64` output to little-endian FP64; a regression test covers FP32
inputs and nonfinite rejection. Use the fresh v2 preparation and run above.
No kernel rebuild or original-reference metadata edit is needed for this fix.

## Validated result (2026-10-23)

Use `prepared-v2/` and `native-v2/`; the failed v1 reference-format experiment
remains separate. All **36/36 final dispatches** complete. For all six layer/slot
cases, private control products match production individually, their FP32 sum
matches captured native down bytewise, and addition reconstructs the native
layer output bytewise. Inactive columns are zero and both candidate placements
match exactly. All six primary projection gates pass with strictly lower
maxrel and rel_fro. Independent CPU re-audit confirms the result.

| Target | Control down maxrel | Candidate down maxrel | Candidate down rel_fro | Composed residual maxrel |
|---|---:|---:|---:|---:|
| L63/P1 | 0.0039805268 | 3.8361005e-6 | 2.9650355e-6 | 0.0070247892 |
| L42/P17 | 0.0020657191 | 2.0427415e-6 | 2.6858477e-6 | 0.0052110173 |
| L52/P17 | 0.0012303268 | 3.4903344e-6 | 2.7144271e-6 | 0.0123485917 |

All composed residuals still fail the unchanged 0.005 bound. L52 worsens from
native **0.0121972127** to **0.0123485917** despite improved local projection:
previous local error partly cancelled incoming drift. This agrees with the
retained ideal-FFN diagnostic. Reports keep `complete=true`, `control_valid=true`,
`projection_passed=true`, `improved=true`, `residual_passed=false` and
`model_acceptance=false`; exit 1 is expected for both run and audit.

Both kernel builds pass all **32-core** resource checks: max text **4784/5568**
bytes (control/low), max L1 buffer end **61448** bytes, configured stack **4096**
bytes, **801 compiler dependencies each**. The reused harness verifies **326**
dependencies and SHA256
`2e83a83f4246d3e45d3bbe74c15b561988c39f59f0f25192566061cddc63ff7b`.

| Artifact | SHA256 |
|---|---|
| Control XCLBIN | 9340da64a21a77d06a0dcc47b4dd9c430c3c347d1ea74ebc21d60b25c7705325 |
| Low XCLBIN | 99e1424192502494d97749231699ce6fde0075b0f7b708d8bf1311f36a2cc527 |
| Both insts.bin | 3648779476135350291ab97dcc6e4fca24cd975cf8efd9bd5f7b4d371d818e26 |

Ten new RED-to-GREEN tests, including explicit FP64 reference storage;
full host suite **1709 passed / 47 skipped**. Model weights, production sources,
manifest and production kernels are unchanged. Four GEMM products replace two
for the tested projection; throughput is unmeasured.

Next replay full-attention fused Q/K/V/gate entry N14336/K5120/T256 and out
N5120/K6144/T256 on the retained causal L63 prefix with exact native controls.
Compose those corrections with the validated down candidate before expanding
private CLI policy or testing fresh full-depth propagation. This stage does
not change the retained **4000/4032** full-depth result or long-prefix status.
