---
name: open-qwen38-l3-projection-control
description: Build and audit current L3 Q/gate/K/V projections with exact native controls and complete INT16/shift/BF16-sum activation-table captures over 240 causal positions.
---

# Current L3 isolated projection control

Use to reproduce the current sequential-prefill attention-entry projections
before diagnosing activation quantization or changing accumulation precision.
The current production export already uses direct FP32 table preparation and
consistent quantized sums. Do not reuse the older L34 original-sum control as
the baseline or interpret its results as current L3 evidence.

## Build and provenance

From the repository root, source `utilities/env.sh`, set
`OPENBLAS_NUM_THREADS=1`, and follow the complete build/resource/prepare/run/audit
commands in `utilities/l3-projection-replay/README.md`. Use fresh directories;
builders and replay refuse existing outputs. Compiler subprocesses and native
NPU execution need the host's normal tool escalation. The retained build is
`Models/qwen38-27b/post-prefill-drift/l3-projection-build-v1`.

`build.py` obtains `dense_prep_f32.cc` and `gemv_q4_gy.cc` from the current
whole-layer generator and copies the current Q4 headers. The private IRON
design consumes five 4096-byte FP32 input elements, runs 224 64-row bands with
40 single-chunk calls per band and emits Q | gate | K | V plus the 11520-byte
table. It uses the production `-Os` functions, 6144-byte stack and unchanged
INT16/shift/high-sum/low-sum table ABI. Table preparation must select both
direct FP32 and `GEMV_Q4_CONSISTENT_SUM=1`.

Keep generated sources, `final.prj`, dependency files, ELF and resource audit.
The current build binds 807 input/dependency files and binary/address hashes.
Measured program text is 3376 bytes; maximum L1 buffer end is 32512. This is
one isolated core's fit, not evidence for an expanded production core.

## Replay contract

The source `post-prefill-drift/native-l3-v1` must have its complete audit and
240 byte-exact full-layer residual controls against current production.
Preparation compares each packed Q/gate/K/V region with independently packed
source tensors, extracts FP32 norms and projections from all 240 captures,
and binds the source/export/compiler/resource evidence. The source q tensor
contains separate Q and gate halves; native output order is Q | gate | K | V.

Use the retained harness in `prefill-sequential-drift/harness-l0-v1/run_kernel`
only with its valid `build-evidence.json`; every dependency and executable
hash is rechecked. All output and table buffers are poisoned for every
dispatch. The same device lock as the native validation runner serializes
hardware access. Logical NPU buffers total 45,964,544 bytes; host RSS is not
measured. The full model is not loaded resident.

Do not edit pinned Python/generated/header files while preparing, executing
or auditing. A report with incomplete coverage, changed hashes, extra captures,
nonfinite outputs or a failed native identity cannot support interpretation.
`control_valid` and `table_valid` are distinct from numerical acceptance.

## Verified checkpoint — 2026-10-28

The 240 dispatches produce **960/960 byte-exact Q/gate/K/V controls** and
**240/240 byte-exact CPU activation tables**, including both BF16 sum planes.
The separate audit rechecks all 480 output artifacts. Preparation binds 8493
inputs and 484 prepared files. Reports stay under
`Models/qwen38-27b/post-prefill-drift/l3-projection-v2` with build, resource,
run, audit and RED/GREEN test logs beside them.

TDD: 10 initial failures; 15 final new tests. Focused suite 49 passed; full
host suite 1988 passed, 47 skipped. Model acceptance remains false. No precision
variant, production change, new throughput claim or full-model fix follows
from an exact control.

Next, use independent FP64 projections of captured norms and reconstructed
INT16 activations to separate signed quantization and accumulation/epilogue
effects. P125/P205 are the main projection-attribution targets. Keep P239 as
a separate attention/cancellation regression control; its remaining attention
component has not been narrowed to softmax or Q normalization alone.
