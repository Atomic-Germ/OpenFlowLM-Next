# Qwen3.8-27B text-only bring-up

Status: in progress. No 27B hardware geometry promoted to the catalogue.

2026-09-24: the user's `LLM_Coding_Agent_Plan.md` supersedes the original
model-only ordering. Current work is shared WideDeltaNet (Track A), then
Qwen3.8-27B (B), then Flash-Next/qwen4exp (C). Current results and next steps
are in [wide-deltanet-bringup.md](wide-deltanet-bringup.md), the
[A7 chain report](wide-deltanet-a7.md), and the
[Track B dense input report](dense-wide-input.md). The phase notes below are
historical; the toolchain is installed, the synthetic A7 chain passes inherited
whole-tensor gates, and Q4 projections at K5120/K6144 pass hardware comparison.

## Baseline (2026-09-23)

- Repository: `c23b1a57f23b6342457b8099ef1f14e3d73f4d47`.
- Initial worktree: only the user's untracked implementation-instructions Markdown.
- Python 3.14.4, NumPy 2.3.5, pytest 9.0.2.
- `ironvenv` absent; `mlir-aie` and `llvm-aie` not installed in host Python.
- XRT 2.26.0 (`e9db9ab15f10173f8d2fc93ff92ab4c7eb09d2e6`),
  amdxdna 2.26.0_20260817, firmware 1.1.2.64.
- Outside the sandbox, `xrt-smi examine` detects Strix aie2p (6x8),
  PCI `0000:67:00.1`, Ryzen AI 9 365. `/dev/accel/accel0` is hidden inside
  the sandbox. Docker is accessible outside the sandbox.
- `python3 -m pytest specs/open-engine/tests/test_qwen35.py -q`:
  **37 passed**.
- `python3 -m pytest specs/open-engine/tests -q`:
  **557 passed, 47 skipped, 2 failed**. Both failures predate changes:
  `test_vision_deepstack.py::test_the_merger_is_the_exact_gelu_not_the_tanh_one`
  and `test_vision_vit_windowed.py::test_the_activations_are_silu_and_the_exact_gelu`
  require missing `scipy.special`. Do not weaken or delete these tests.

## Phase 1: official geometry

Fixture is the unchanged official `config.json` from
`https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json`.
SHA256: `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`.
The fixture retains vision metadata for provenance; only its text tower is used.

The existing parser already derives `qwen35`, hidden 5120, FFN 17408,
64 layers, 24Q/4KV, head dim 256, rotary dim 64, 16 key / 48 value
DeltaNet heads of dimension 128, QKV width 10240, value width 6144,
vocabulary 248320, and `(linear, linear, linear, full) * 16`.
No architecture or parser change is needed.

TDD: added the geometry tests first (2 fixture-missing errors), then installed
the pinned fixture. Geometry plus existing Qwen3.5 tests: **39 passed**.
Hardware: **NOT VALIDATED**. Next: capture resource blockers as tests.

## Phase 2: resource blockers

`test_qwen35_27b.py` captures physical constraints without relaxing them:

- xn needs three 4096-byte chunks, carrying 32 / 32 / 16 AB weight tiles.
- The old one-bank shared-side schedule alone needs 14 fills, exceeding 13.
- Full FFN activation table: 39168 bytes. With `PER_CALL=1`, legacy scratch,
  stream buffers and stack total 69888 bytes, exceeding 61440.
- 48 value heads require two 32-lane banks with 32 / 16 active lanes.
- New primitive points remain refused without unvalidated mode.

Tests: **6 passed** before packing work. Hardware: **NOT VALIDATED**.

## Phase 3: bank-major AB packing

Added `transpose_banked` to both pack interpreters and the C++ manifest
parser. Qwen3.5 selects it only when `lin_value_heads > 32`.
Source remains `[heads, hidden]`; destination is `[banks, hidden, 32]`.
Bank 0 holds heads 0..31; bank 1 holds heads 32..47 plus 16 zero lanes.
Each bank at hidden 5120 is 327680 bytes, or 80 4096-byte weight tiles.
Existing `transpose` and all <=32-head plans retain their prior layout.

TDD red results: 9 new Python failures for the absent op/plan selection;
2 C++ pool failures for unknown op; 5 C++ manifest failures for unknown op
and missing-field diagnostics. Green results:

- Qwen3.5 and 27B tests: **52 passed**.
- Whole open-engine: **572 passed, 47 skipped, 2 failed**; same two missing
  SciPy failures as the baseline, no new failures.
- C++ pool tests: **PASS**, including full-width synthetic BF16 storage,
  byte/element order, zero padding, untouched canaries and invalid inputs.
- C++ manifest tests: **PASS** on all six checked-in family fixtures.
- Pool tests and Qwen3/Qwen3.6 manifest tests also **PASS in Docker** using
  the existing `q38rocm-qwen38-server:latest` image, ID `e7648c3e83b4`.
  These are CPU packing/schema tests, not NPU numerical comparisons.

Reproduce from repository root:

```bash
python3 -m pytest specs/open-engine/tests/test_qwen35.py specs/open-engine/tests/test_qwen35_27b.py -q
python3 -m pytest specs/open-engine/tests -q --tb=short
g++ -std=c++17 -O2 -Isrc -Isrc/include -Iopen_kernels/harness \
  src/open_qwen36/pools_test.cpp src/open_qwen36/pools.cpp \
  src/open_qwen36/q4nx_file.cpp -o /tmp/oflm-pools-test
/tmp/oflm-pools-test
g++ -std=c++17 -O2 -Isrc -Isrc/include -Iopen_kernels/harness \
  src/open_qwen36/manifest_test.cpp src/open_qwen36/manifest.cpp \
  -o /tmp/oflm-manifest-test
/tmp/oflm-manifest-test \
  specs/open-engine/tests/fixtures/manifest_qwen36.json \
  specs/open-engine/tests/fixtures/manifest_qwen3_4b.json \
  specs/open-engine/tests/fixtures/manifest_gemma3_4b.json \
  specs/open-engine/tests/fixtures/manifest_hy_mt2_7b.json \
  specs/open-engine/tests/fixtures/manifest_qwen35_9b.json \
  specs/open-engine/tests/fixtures/manifest_phi4_mini_4b.json
docker run --rm --network none --read-only --tmpfs /tmp \
  --mount type=bind,src=/tmp/oflm-pools-test,dst=/tests/pools,readonly \
  --entrypoint /tests/pools q38rocm-qwen38-server:latest
```

The full 27B plan still fails its L1 gate. The isolated AB plan-selection test
uses the exact attention geometry with a smaller FFN to exercise packing
without bypassing that gate. No manifest advertised for 27B, no catalogue or
`MIXED_CORE_FITS` changes, no new engine family. Recipe source hashes will
change as designed; this does not imply any existing generated layout changed.

## Phase 4: wide glue worker (2026-09-23)

Implemented the geometry-gated wide worker in `designs/layer_x/lx.py`:

- `ab_banks(spec) = ceil(value_heads / 32)` in the existing shared recipe.
- `xn_side` ObjectFifo exists only for the dense wide path, with depth 1.
  It is appended to worker/runtime handles only for that path. Physical shim
  placement is intentionally left to phase 5; no per-FIFO shim allowance is assumed.
- For each bank: alpha's 3 xn chunks and 80 weight tiles, beta's same replay,
  then small parameters. Two `acc[32]` buffers are reused. Decay/beta each hold
  48 floats. xn is copied into the existing private chunk and released before
  weight consumption, so it needs only one FIFO slot.
- The existing conv and record-emission loop follows both banks unchanged.
- `glue_small_bank.h/.cc` provides the bank-aware helper needed by this worker
  (the addressing part of phase 6). A is read at `base+h`, dt_bias at
  `NHEAD+base+h`, and the accumulators at `h`; the final bank handles 16 lanes.
  The existing `dn_glue.h`, `glue_small.cc`, `glue_ab_e.cc` and `dnx.h` are untouched.
- A wide recipe with a small enough FFN now fails explicitly with
  `not implemented: wide DeltaNet glue DMA scheduling`, including in
  unvalidated mode. The host sequence has a second explicit guard. The real
  17408 FFN still hits the earlier L1 gate. Neither route silently uses the
  legacy host sequence with the new consumer order.

Declared glue-core storage (not the main-core segmented FFN budget):

| Allocation | Bytes |
|---|---:|
| side, depth 2 | 8192 |
| xn_side, depth 1 | 4096 |
| gact, depth 5 | 10240 |
| gout, depth 3 | 6144 |
| acc_a / acc_b, 32 floats each | 256 |
| decay / beta, 48 floats each | 384 |
| qk / vt / xnb | 24576 |
| stack | 6144 |
| **Total declared** | **60032** |

Declared headroom against the 61440-byte recipe budget is 1408 bytes. A
depth-2 xn FIFO would raise the total to 64128. This sums source declarations;
IRON alignment, bookkeeping, placement and program memory are **NOT VALIDATED**.
The generated allocation report in phase 5 must take precedence.

TDD: before implementation, the new tests gave **4 failures / 4 passes**:
missing wide input, missing bank geometry, missing explicit DMA guard, and
missing C++ helper. After implementation and expanded regression checks:

- `test_qwen35_wide_glue.py`: **14 passed**. Tests execute the actual worker
  body, check stream depletion/acquire-release balance, exact integer-data
  GEMV sums, reused accumulator identities, both banks, and all 48 records.
  Legacy worker cases cover 0.8B / 2B / 4B / 9B and the MoE path. Type/FIFO
  declaration tests cover legacy topology, wide widths and declared L1.
- Existing Qwen3.5 + 27B + wide-glue tests: **66 passed**.
- Full open-engine: **586 passed, 47 skipped, 2 failed**. The same two
  pre-existing missing-SciPy vision failures; no new failures.
- The production small-helper header compiles and runs with host scalar math
  under GCC undefined/bounds sanitizers, checking all heads, both A/dt_bias
  arrays, inactive accumulator NaNs and output canaries. This is not a test
  of AIE vecmath approximation error.
- The standalone C++ helper test also **PASS in Docker**.

Reproduce:

```bash
python3 -m pytest specs/open-engine/tests/test_qwen35_wide_glue.py \
  specs/open-engine/tests/test_qwen35.py specs/open-engine/tests/test_qwen35_27b.py -q
g++ -std=c++17 -O2 -Wall -Wextra -Iopen_kernels/designs/dn_glue \
  specs/open-engine/tests/fixtures/glue_small_bank_test.cpp -o /tmp/oflm-glue-small-bank-test
/tmp/oflm-glue-small-bank-test
docker run --rm --network none --read-only --tmpfs /tmp \
  --mount type=bind,src=/tmp/oflm-glue-small-bank-test,dst=/tests/glue-small-bank,readonly \
  --entrypoint /tests/glue-small-bank q38rocm-qwen38-server:latest
```

Files changed: shared recipe bank count, qwen35 dispatch guard, `lx.py`, new
`dn_glue/glue_small_bank.h/.cc`, `tests/test_qwen35_wide_glue.py`,
`tests/fixtures/glue_small_bank_test.cpp`, canonical spec and this report.
No xclbins/libraries were built. Source-based cache keys change intentionally;
legacy layout fixtures and interpreted worker behavior still pass, but binary
identity has not been established by a rebuild.

## Outstanding gates after phase 4 (historical; superseded by Track A report)

Phase 5: implement and compile/place the wide host schedule (7 logical weight
fills and 12 xn fills). The temporary wide-dispatch guards above must be
replaced only with a working bring-up schedule. Prepare IRON/Peano first;
the repository's `ironvenv` is still absent. To isolate glue before segmented
FFN exists, use a dedicated glue design or an explicitly synthetic narrower
FFN geometry, never label that the full 27B build. Capture actual shim/L1
diagnostics and preserve all legacy limits. Establish scheduling feasibility
before hardware execution; CPU tests do not validate physical resources.

The remaining phases are not implemented: host DMA, segmented
FFN, primitive and real-layer numerical bring-up, converter registration,
model packaging and chat checks. In particular the GEMV accumulator is in
even/odd lane order between tiles and only zipped to row order on `last`;
segment boundaries must preserve that invariant, not just clear `first`.
The planned down segments are 8192 / 8192 / 1024; no full K=17408 table.

Hardware comparisons, 8-layer and 64-layer multi-token inference, prefill /
decode timings and the LLM functional suite: **NOT RUN**. No xclbins,
libraries or model packages were built; no kernel/build skill is claimed.
Exact model conversion, kernel build and inference commands remain to be
established by the corresponding later phases, not advertised as working.
