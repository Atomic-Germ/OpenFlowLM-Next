# Qwen3.8-27B text-only bring-up

Status: in progress. No 27B hardware geometry promoted to the catalogue.

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

## Next work and outstanding gates

Phase 4 starts with a dedicated xn ObjectFifo on the wide path. Keep AB
accumulators at 32 floats; decay/beta need storage for the real 48 heads.
The bank-aware small helper must index A at `base+h` and dt_bias at
`NHEAD+base+h`, while indexing the reused accumulator at `h`.
Hardware scheduling must be compiled and placed before interpreting the
logical 7 weight / 12 xn fills as physically feasible.

The remaining phases are not implemented: bank-aware glue/DMA, segmented
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
