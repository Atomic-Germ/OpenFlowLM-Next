# Isolated paired-weight QKV/Z replay

Use captured L34/P6 and L37/P0 from the retained 20-row prefixes. These exposed
Q head 13 (0.0050724904) and V head 46 (0.0071224278) failures under the unchanged
0.005 bound. CPU BF16-weight predictions fit closely; this experiment tests a
real NPU precision variant with byte-exact production/native controls first.

## Build contract and commands

Build private **N16384/K5120/T256, Q4_1, no B reuse** copies of the production
design/header. Control preserves all arithmetic. Low changes only fused Q4
dequant conversion to BF16_RNE(W - BF16_RNE(W)) via accfloat widening. Reuse the
validated helper from the FFN experiment, without modifying its pinned sources.
Sanitize all GQP flags, reject existing output directories and record source,
generated, compiler dependency and binary hashes. Audit all 32 core text/L1
budgets and retain failed projects. No production source or ABI changes.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-entry-replay/build.py \
  --mode control --out Models/qwen38-27b/prefill-entry-replay/builds/control
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-entry-replay/build.py \
  --mode low --out Models/qwen38-27b/prefill-entry-replay/builds/low
```

The builder selects the installed ironvenv mlir-aie and bundled Peano and sets
its own PATH. Kernel compilation requires host access in this environment.
Reuse the already verified standard harness; do not rebuild it unnecessarily.
Document/register the associated skill before building this new kernel set.

## Prepare and replay

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-entry-replay/check.py prepare \
  --out Models/qwen38-27b/prefill-entry-replay/prepared \
  --attribution Models/qwen38-27b/prefill-projection-attribution/run \
  --capture Models/qwen38-27b/prefill-prefix-sublayers/run \
  --capture-build Models/qwen38-27b/prefill-prefix-sublayers/build \
  --baseline Models/qwen38-27b/prefill-weighted-cli/v2/candidate/run
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
ironvenv/bin/python utilities/prefill-entry-replay/check.py run \
  --out Models/qwen38-27b/prefill-entry-replay/run \
  --prepared Models/qwen38-27b/prefill-entry-replay/prepared \
  --build-root Models/qwen38-27b/prefill-entry-replay/builds \
  --harness-build Models/qwen38-27b/prefill-upgate-replay/harness
```

Preparation revalidates native/compiler/kernel/model/reference provenance and
CPU attribution artifacts. Use xn/QKV/Z row 6 from `hybrid_L34_P17_*` and row 0
from `hybrid_L37_P7_*`. Repack fused weights from the manifest, require exact
source/effective identities and match their retained hashes. Primary references
remain original FP32-input/effective-weight FP64 arrays from the CPU stage;
paired-input references are diagnostic only.

Run serially with exclusive NPU access. No full-model residency: one process
uses 52428800 weight bytes, 2621440 input bytes, 16777216 output bytes (~68.5 MiB),
excluding XRT overhead. Placements are 6/134 and 0/128; all other columns must be
zero in **every** individual product, not just the combined output.

There are 24 dispatches: two target rows × two placements × production/control/
low-weight × high/low input. Production sums must reproduce concatenated native
QKV/Z captures bytewise. New control must reproduce **each individual** production
product bytewise, so cancellation cannot conceal a changed control. Compare the
candidate at both placements bytewise.

Candidate combination uses FP32((WhXh + WhXl) + (WlXh + WlXl)), preserving the
validated host sum order. Require all 128 Q/K/V/Z head gates per case (512 total)
to pass maxrel <0.005 and rel_fro <=0.005 against the primary reference, with
strict improvement in both errors for every head. Preserve numerical failures,
not just control failures. Exit zero denotes isolated projection success;
`model_acceptance` is always false. No corrected convolution, recurrent state,
residual, FFN or logits are produced by this replay. Four products replace two;
no throughput claim. Full-depth/long-prefix accuracy requires later validation.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q \
  specs/open-engine/tests/test_prefill_entry_replay.py
```

Tests cover geometry/environment isolation, existing evidence protection,
conversion-only patching, individual/summed native controls, inactive columns,
strict per-head reference gates, FP32 sum order and persisted failure status.

## Measured result — 2026-10-21

Artifacts: `Models/qwen38-27b/prefill-entry-replay/`. Both builds and all 24
serial dispatches pass. Native sums, individual production/control products,
zero inactive columns and candidate slot identity pass. **512/512 primary head
checks** (256 distinct heads × two placements) pass and strictly improve both
errors. Q L34/P6/H13 maxrel: **0.0050724904 → 2.8237745e-6**;
V L37/P0/H46: **0.0071224278 → 1.4844200e-5**. Worst candidate maxrel/rel_fro:
**1.4844200e-5 / 7.8625458e-6**. No corrected causal state or model outputs.

mlir-aie **1.4.2**, llvm-aie **21.0.0.2026080301+c9c5ecb7**; 801 pinned compiler
dependencies per build. All 32 cores pass budgets; maximum text control/low:
4784/5568 bytes, maximum L1 buffer end 61448 bytes. XCLBIN SHA256:

- control: `dfa3e2c443fdd53fcea9a6605d137058eb0d90e02a11da2a3fdf45f16ec7f125`
- low: `e3a8e68eca6747f3cb437ce896d7e30690cbb3f511bbc43e955118f079104fa8`

Both `insts.bin`: `35c90e689c5f179a8430ba9a2e7744f8e2f3ff7eaa83de528b1d8092e04d7d35`.
12 new tests pass after initial RED; full host suite **1604 passed / 47 skipped**.
Full retained model residual acceptance remains **3974/4032 FAIL**. Next test
complete captured prefixes through causal host state and output projection.
