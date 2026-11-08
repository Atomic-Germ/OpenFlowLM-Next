# Private integrated L3 projection refinement

This experiment moves the audited two-pass projection correction into a private
full-ax kernel. It derives the quantization residual from the native FP32 entry
norm, prepares high/low INT16 tables with production consistent sums, evaluates
both with the same production Q4 GEMV entry and adds the completed bands in FP32
on the NPU. Subtraction and final addition use private integer IEEE binary32
round-to-nearest-even helpers to reproduce host results exactly. Each weight element is consumed once and used for both passes.
No host projection sum or external refined attention input is supplied.

Only the Q/gate/K/V entry changes. Output projection, FFN, normalization and
attention workers retain production arithmetic. Two tables share an enlarged
23,040-byte preparation buffer; the second table is dead before output/FFN
preparation reuses it. A separate 64-float buffer retains the low band. The
original six runtime buffers, scratch layout, DMA sequence and position-table
patching contract are retained. Production files are not edited.

## Reproduce

Require the independently audited 240-position forced-tail sweep and all its
pinned upstream evidence. Use fresh output directories and the existing audited
harness. No full-model residency is needed: only the L3 weight pool is loaded.

```bash
source utilities/env.sh
export OPENBLAS_NUM_THREADS=1
ironvenv/bin/python utilities/l3-refined-ax/build.py --role control \
  --spec Models/qwen38-27b/production-post-fp32/kernels-v1/spec.json \
  --out Models/qwen38-27b/post-prefill-drift/l3-refined-ax-control-v2
ironvenv/bin/python utilities/l3-refined-ax/build.py --role candidate \
  --spec Models/qwen38-27b/production-post-fp32/kernels-v1/spec.json \
  --out Models/qwen38-27b/post-prefill-drift/l3-refined-ax-candidate-v2
ironvenv/bin/python utilities/l3-refined-ax/replay.py prepare \
  --baseline Models/qwen38-27b/post-prefill-drift/l3-tail-sweep-v1 \
  --control-build Models/qwen38-27b/post-prefill-drift/l3-refined-ax-control-v2 \
  --candidate-build Models/qwen38-27b/post-prefill-drift/l3-refined-ax-candidate-v2 \
  --out Models/qwen38-27b/post-prefill-drift/l3-refined-ax-v2
ironvenv/bin/python utilities/l3-refined-ax/replay.py run \
  --out Models/qwen38-27b/post-prefill-drift/l3-refined-ax-v2 \
  --harness Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1/run_kernel
ironvenv/bin/python utilities/l3-refined-ax/replay.py audit \
  --out Models/qwen38-27b/post-prefill-drift/l3-refined-ax-v2
```

## Acceptance of this experiment

- Build the control separately. Pin actual compiler dependencies, copied
  generated translation units, ELFs, allocated MLIR and binary hashes. Reject
  changed render anchors, unsupported geometry and swapped build roles.
- Run each role in a separate harness invocation under the exclusive device
  lock. Initialize its entire KV buffer once at position zero, then retain its
  own causal KV through all 240 positions. Original layer-input residuals and
  preceding scratch are loaded each position; candidate residuals are not fed
  into another layer or another case.
- Require all 720 original control captures bytewise before interpreting the
  candidate. Capture inventories must contain exactly 1440 files.
- Require all 960 candidate projection vectors to match the staged two-pass
  experiment bytewise. Compare the complete candidate scratch to the audited
  forced-tail scratch with only Q/gate/K/V/OG replaced by the audited refined
  values. Require all 240 candidate KV prefixes and residuals bytewise too.
- Recompute residual metrics against retained independent FP64 local references,
  conditioned-tail references and full-model CPU residuals, and attention-head
  metrics against retained independent primary/conditioned attention. Require
  the retained measurements and regression summary to match. The output/FFN
  sublayer gates are transferred through complete scratch/residual identity to
  the audited sweep; they are not a new weight-dequantization audit.
- Audit the saved captures in a separate invocation. Source, baseline evidence,
  build roles, harness, logs and artifacts must remain bound. Keep numerical
  acceptance separate from execution/audit completion. No tolerance changes.

Full-model propagation, production performance and resident request isolation
remain separate gates. A byte-exact match preserves the staged experiment's
regressions as well as its improvements; it does not prove uniform improvement.

## Rounding diagnostic

The first integrated candidate failed byte identity. An isolated probe reuses
its prep/sum helper sources and separates the two operations without a weight
pool. The original vector implementation matched 240/240 high tables, only
162/240 low tables, and only 2/240 complete sums (1532 differing sum elements).
Cancellation can amplify a small operand error to several result ULPs. Retain
that failed run and its source snapshots; do not weaken identity checks.

`add_rne.h` implements binary32 addition in integer arithmetic, including guard,
round and sticky bits, cancellation, subnormals and signed zero. It is used
both for the entry quantization residual and the final projection sum. Host
checks compare random finite operands, subnormals and selected cancellation/
rounding cases against the host FPU. All arithmetic still runs on the NPU.

```bash
ironvenv/bin/python utilities/l3-refined-ax/probe.py \
  --out Models/qwen38-27b/post-prefill-drift/l3-refined-probe-v4
ironvenv/bin/python utilities/l3-refined-ax/diagnose.py \
  --build Models/qwen38-27b/post-prefill-drift/l3-refined-probe-v4 \
  --source Models/qwen38-27b/post-prefill-drift/l3-refinement-v1 \
  --out Models/qwen38-27b/post-prefill-drift/l3-refined-diagnosis-v2 \
  --harness Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1/run_kernel
ironvenv/bin/python utilities/l3-refined-ax/diagnose.py --audit \
  --out Models/qwen38-27b/post-prefill-drift/l3-refined-diagnosis-v2
```

The probe packs high/low bands into one FIFO, retaining only two input DMA
channels (entry norm and projection pairs). Do not give high and low separate
input FIFOs: that exceeds the core's two-channel limit. It also needs the
production `gemv_q4` header include path. Both unsuccessful probe builds are
retained under `l3-refined-probe-v1/v2`; v3 diagnoses the original rounding,
v4 validates the corrected helpers.

## Measured result (2026-10-29)

The corrected probe matches **240/240 high tables, 240/240 low tables and
240/240 sums**, with zero differing sum elements. Its separate audit binds all
480 captures. One-core resources: .text 4400 B, maximum L1 end 34,304 B,
stack 6144 B; 813 pinned compiler inputs.

Fresh full-ax control/candidate builds complete **480 NPU dispatches**. All
**720 native control captures** match bytewise. The integrated candidate matches
all **960 refined projections**, **240 complete expected scratch buffers**,
**240 causal KV prefixes** and **240 output residuals** exactly. Each candidate
request uses its own causal KV from position zero, with no host correction.
Separate audit verifies all 1440 captures and recomputes residual/head metrics.

All 5760 candidate primary and conditioned attention-head gates pass. All 240
residual cases pass and reproduce the staged numerical results: primary maxrel
1.896555707e-5, 178 improved / 61 regressed / one tied; full-model-reference
maxrel 3.757162146e-5, 155 improved / 83 regressed / two tied. Integration does
not erase the retained regressions or improve on the staged candidate.

Both 15-core builds retain an identical instruction stream. Main-core .text
is 9920 B for the control and 12,528 B for the candidate; main-core L1 ends
are 51,456 / 54,016 B. Overall maximum .text remains 13,760 / 16,384 B, maximum
L1 end 59,920 / 65,536 B, all stacks 6144 B. Control/candidate bind 858/868
compiler inputs. No production kernel or installed runtime was replaced.

TDD: 14 new tests, including native-compiled integer arithmetic against random
finite FP32 operands, subnormals, cancellation and signed zero. Full host suite:
**2057 passed, 47 skipped**. Failed initial builds/runs and the vector-helper
source snapshot are retained separately. Next connect the private candidate to
a separate runtime build and validate propagation through all 64 layers with
the existing independent references. Full-model numerical acceptance remains
**FAIL**; these instrumented runs do not establish production performance.
