# Composed FFN replay on captured causal input

This diagnostic composes the validated paired-weight L63/P7 up/gate result
with the production host activation and unchanged paired-input down GEMM.
It starts from the retained pre-FFN residual and norm, not an artificial
embedding or CPU replacement for native state. It does not replay attention.

A private CPU bridge extracts the current RMSNorm function and FFN activation
expression from block_host.cpp/core.cpp, preserving their arithmetic. It must
reproduce captured norm and control activation bytewise. Record source,
generated code, compiler dependencies and shared-library hashes before reuse.
No production source or kernel changes are needed.

Use fresh directories and the existing paired-weight replay/harness artifacts.
The reproduction commands are:

```bash
ironvenv/bin/python utilities/prefill-ffn-replay/bridge.py \
  --out Models/qwen38-27b/prefill-ffn-replay/bridge
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-ffn-replay/check.py prepare \
  --out Models/qwen38-27b/prefill-ffn-replay/run \
  --bridge Models/qwen38-27b/prefill-ffn-replay/bridge
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
ironvenv/bin/python utilities/prefill-ffn-replay/check.py run \
  --out Models/qwen38-27b/prefill-ffn-replay/run \
  --bridge Models/qwen38-27b/prefill-ffn-replay/bridge
```

The fixture paths default to the retained project-local artifacts documented
in the paired-weight runbook. No model download or full-model residency.
Down shape is N5120/K17408/T256, with the captured row placed in slots 7/135.
Control activation, down and final residual must match retained native bytes.
Each individual high/low down output must have zero inactive columns; candidate
slots must agree bytewise. Pack down from the manifest and prove equality to
the hashed production pool and effective source weights before dispatch.

Keep two gates separate: conditioned FFN against ideal math on the actual
pre-FFN residual, and final layer output against the retained independent CPU
reference. Both require maxrel <0.005 and rel_fro <=0.005. The maxrel bound is unchanged;
the Frobenius gate is additional to the earlier residual-only comparison. Up/down cancellation
can make a more accurate projection worsen the layer result. A passing isolated
FFN does not establish accuracy on other rows, causal state or full logits.

No new XCLBIN. Run eight down dispatches serially with exclusive NPU access;
all source/build/input/capture hashes must validate before interpretation.
Failed identity or execution must invalidate the report.

## Measured result (2026-10-20)

All **8/8 down dispatches** complete. Control norm, activation, down and final
residual reproduce the retained native bytes. Candidate slots 7/135 agree
bytewise, and each high/low output has zero inactive columns. Down packing
matches the retained pool bytewise and effective source weights elementwise.

| Gate | Control maxrel | Candidate maxrel | Candidate rel_fro |
|---|---:|---:|---:|
| Conditioned FFN | 0.0005411159 | 0.0002390673 | 0.0010533278 |
| Independent layer output | 0.0058400461 | 0.0045267063 | 0.0023137023 |

The candidate crosses the original layer-output threshold on this captured
row. This is a composed native FFN result with a retained causal input, not a
fresh causal model run. The earlier full-depth long-prefix accuracy failures
remain open. Expand selected-row coverage before integration and repeat
causal propagation gates after integration.

The CPU bridge uses GCC 15.2.0, 105 compiler dependencies and source/generated
code hashes. Bridge SHA256:
`a3c18dbd7ec7bc3cad75bd9339e922cae923f59fbdd5f6106b6ac06df54f3ca3`.
The unchanged production down and prior paired-weight builds/harness are
verified before reuse. Audit/report JSON files pin all inputs and outputs.
Evidence lives under `Models/qwen38-27b/prefill-ffn-replay/` (ignored).

TDD: 9 new tests initially fail (including missing-module fixture errors),
then pass. Full host suite: **1531 passed, 47 skipped**. No new XCLBIN or
production arithmetic change; only one projection's weights are dispatched.

## Selected-row expansion

`sweep.py` reuses the same bridge, harness and private GEMM builds. It selects
linear (L60/P38) and attention (L35/P80) manifest pool offsets independently.
Preparation verifies captured host arithmetic, packed bytes and effective
weights for both projections. Exact-weight FP64 up/gate references retain the
original FP32 norm; the independent conditioned FFN and global layer references
remain separate. No new build or full-model residency is required.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-ffn-replay/sweep.py prepare \
  --out Models/qwen38-27b/prefill-ffn-sweep/run
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
ironvenv/bin/python utilities/prefill-ffn-replay/sweep.py run \
  --out Models/qwen38-27b/prefill-ffn-sweep/run
```

Use exclusive NPU access. Forty serial dispatches test each original slot and
its counterpart in the other T256 half (38/166 and 80/208). Fresh production
and private-control up/gate high/low products must agree individually, and
control norm/H/down/output must match the native captures bytewise. Candidate
up/gate/H/down/output must be byte-identical across placements. Each raw dump
must be complete, finite and zero outside the selected column. Reports retain
per-row local and global failures; a numerically failing completed experiment
returns 1. Execution/provenance/identity failures invalidate the entire control.
The sweep does not substitute ideal state or claim causal propagation.

### Selected-row results (2026-10-21)

All **40/40 dispatches** complete. Both production/private high and low
controls agree individually. Norm, H, down and output match retained native
captures; all candidate boundaries agree across slots, with zero inactive
columns. Both linear and attention pool/source-weight checks pass.

| Row | Candidate up/gate maxrel | Control → candidate FFN maxrel | Control → candidate output maxrel |
|---|---|---|---|
| L60/P38 | 1.7964e-6 / 1.6948e-6 | 0.0009571130 → 0.0008939331 | 0.0014798680 → 0.0014800810 |
| L35/P80 | 2.2408e-6 / 1.8304e-6 | 0.0016492574 → 0.0005499310 | 0.0001922195 → 0.0001962610 |

Both local and final-output gates pass (maxrel <0.005 and rel_fro <=0.005).
Final-output Frobenius errors improve to 0.0046959671 / 0.0025036066, but
maximum errors increase slightly. Thus improved FFN math does not guarantee
monotonic final-output improvement. These hybrid rows already passed globally;
this does not fix the sequential route's failures at those positions.

TDD: **14 new tests fail before implementation, then pass; 1545 passed,
47 skipped** overall. Artifacts and provenance are in
`Models/qwen38-27b/prefill-ffn-sweep/`. All builds are reused unchanged.
Next use a private native candidate to test fresh causal propagation with
exact controls before considering production integration. Full-model
long-prefix acceptance remains unresolved.
