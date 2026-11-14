---
name: open-qwen38-prefill-host-boundaries
description: Build a CPU bridge to production Qwen3.8 prefill host stages and validate FP32 conv history, paired-BF16 KV, padding and 255/256/257-token boundaries against an independent FP64 reference.
---

# Qwen3.8 host prefill boundary validation

Use for conditioned CPU-stage checks before full-model prefill. This does not
establish NPU projection accuracy or complete prefill support. No model weights,
XRT context, kernel rebuild or full-model memory reservation is needed.

## Build and run

From the repository root on Linux x86_64 with AVX2, g++, OpenMP and NumPy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
  ironvenv/bin/python utilities/prefill-boundaries/check.py \
  --out Models/qwen38-27b/prefill-host-boundaries/wide
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
  ironvenv/bin/python -m pytest -q specs/open-engine/tests/test_prefill_boundaries.py \
  specs/open-engine/tests/test_prefill_precision_reference.py \
  specs/open-engine/tests/test_prefill_batch.py
```

`--small` selects a lightweight 3:1 linear-head fixture. The default derives
geometry from `open_kernels/recipes/specs/qwen35-27b.json`, including recipe
AB_LANES=64 and S_ROWS=130. It uses H5120, 16/48 linear heads of dimension 128,
four conv taps, 24/4 attention heads of dimension 256 and rotary dimension 64.

The script always builds `bridge.cpp` and production `block_host.cpp` into a
private shared library. It uses `-O2 -mavx2 -mfma -ffp-contract=off -fopenmp`.
Do not remove contraction control: partition identity depends on using the
production arithmetic. `build.json` includes exact commands, compiler version,
local source/header hashes from compiler `.d` dependencies and the library hash.
No production C++ arithmetic is copied into the bridge.

## Acceptance and evidence

Lengths 1/2/255/256/257 each receive one additional continuation token. Nonzero
initial state and a three-row KV prefix exercise carry; a nonzero cache tail
exposes spurious writes. Padded inputs contain NaN; padded outputs and S rows
must remain zero. Conv history is FP32; cache planes are K hi/lo, V hi/lo.

Require byte-exact native block/sequential results for outputs, conv, S and KV;
exact conv versus reference; unchanged original prefix and unwritten tail after
each block; independent numerical gates for DeltaNet/attention outputs, final S
and reconstructed K/V. Each max absolute error normalized by max reference
magnitude must be below 1e-3. Never accept native/native parity alone.

The NumPy oracle promotes input operands to FP64 before arithmetic, stores only
the specified state/cache boundaries at their target precision, and never reads
native intermediate state. All fixtures/results are retained as NPZ files.
`report.json` pins their hashes, source/spec hashes and the build record.
A rerun invalidates prior success before compiling; failures remain failed.

2026-10-18: five wide and five small cases pass. Wide worst errors:
DeltaNet 3.9749723e-7, attention 8.8448682e-7, S 2.5862322e-7, cache 6.1066092e-6.
All partition and history identities pass. Targeted suite 34 passed, including
legacy BF16 coverage and tests that reject matching-but-wrong outputs, cache
low-plane loss, padding/history corruption and stale success after build failure.

Next validation must include actual projection/GEMM results and full-model
prefill routes. Paired-KV block attention currently runs on the host; do not
report these CPU checks as fully NPU prefill or as end-to-end model acceptance.
