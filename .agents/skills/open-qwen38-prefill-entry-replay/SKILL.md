---
name: open-qwen38-prefill-entry-replay
description: Build private paired-weight QKV/Z entry kernels and replay L34/P6 and L37/P0 with exact native controls, per-head gates and two slot placements without full-model residency.
---

# Isolated QKV/Z precision

Read [the runbook](../../../utilities/prefill-entry-replay/README.md) before
building. Register this workflow first. Use private copies of the production
GEMM design/header at N16384/K5120/T256, Q4_1, no B reuse. Control is unchanged;
low-weight changes only the fused dequant BF16 conversion, using the validated
accfloat widening helper. Do not modify pinned FFN helpers or production files.
Sanitize GQP flags, reject existing directories and pin generated sources,
compiler dependencies, binaries and all 32 core resource checks. Retain failures.

Use captured row 6 from L34/P17 and row 0 from L37/P7 prefix files. Revalidate
attribution/native/build/model provenance and exact packed/source weights.
Keep original-input effective-weight FP64 references primary. Run 24 serial
NPU dispatches with exclusive access using the existing verified harness;
no full model residency is required (~68.5 MiB logical BOs per dispatch).

Require exact production paired-input sums against concatenated native QKV/Z,
then exact individual new-control products against production. Every product
must have finite full output and zero inactive columns. Candidate FP32 sum
order is (WhXh+WhXl)+(WlXh+WlXl); placements 6/134 and 0/128 must match exactly.
All 512 primary head gates must pass unchanged maxrel/rel_fro bounds and strictly
improve both errors. An improved failing candidate is still FAIL. Preserve
failure reports and hashes. Isolated projection PASS is never full-model or
causal-state acceptance; four products replace two and throughput is unmeasured.

## Validated checkpoint — 2026-10-21

Artifacts under `Models/qwen38-27b/prefill-entry-replay/`: both builds and
24/24 serial dispatches pass. Exact native/individual controls, zero inactive
columns and slot identity pass; 512/512 primary head checks strictly improve.
Q L34/P6/H13 maxrel 0.0050724904 → 2.8237745e-6; V L37/P0/H46
0.0071224278 → 1.4844200e-5. Worst candidate rel_fro 7.8625458e-6.
mlir-aie 1.4.2 / llvm-aie 21.0.0.2026080301+c9c5ecb7; 801 compiler deps/build;
32-core max text 4784/5568 bytes, L1 buffer end 61448 bytes. Full hashes and
commands are in the runbook/build evidence. 12 RED-to-GREEN tests; full suite
1604 passed / 47 skipped. Model residual acceptance remains 3974/4032 FAIL.
Next extend to complete captured prefixes and causal host/out composition;
these two isolated rows cannot establish propagated state or model accuracy.

## Full-prefix follow-up — 2026-10-22

Use [the prefix replay runbook](../../../utilities/prefill-prefix-replay/README.md)
for composition across all 20 captured rows of L34/L37/L53. Reuse these entry
kernels, original output kernels and the verified host bridge; no rebuild.
Derive host alpha/beta lanes from real tensors (48), never padded recipe lanes
(64), and validate all constant sizes before the C ABI. Require full native
host output/conv/state byte identity before interpreting candidate precision.

`Models/qwen38-27b/prefill-prefix-replay/v2/`: 60 dispatches, 15360 entry checks,
6048 conditioned host checks and 240 output projection checks pass. All controls,
slot placements and block/sequential host calls agree bytewise. Candidate
pre-FFN residual maxrel L34/P17 0.0006256207, L37/P7 0.0008810181,
L53/P17 0.0074347548: the last still FAILS. 14 new RED-to-GREEN tests;
1618 passed / 47 skipped overall. Initial wrong-stride attempt was rejected
before NPU dispatch and remains recorded under `run/`. Next isolate paired-weight
output projection; do not equate local prefix improvements with model acceptance.
