---
name: open-qwen38-prefill-paired-weights
description: Build private high/low dequantized-weight GEMM kernels and validate Qwen3.8 causal up/gate precision against exact native controls without model residency.
---

# Private paired-weight prefill GEMM

Read the [runbook](../../../utilities/prefill-paired-weights/README.md) before
building. This experiment follows the verified L63/P7 up/gate replay: effective
packing is exact, but the BF16 weight boundary accounts for most of its error.
Keep production design, kernels and Q4NX unchanged. The builder copies the
existing design/header and changes only the low-weight header conversion.

Control emits BF16_RNE(W). Low emits BF16_RNE(W-BF16_RNE(W)) from the same FP32
Q4 dequant accumulator. Use separate unchanged-layout GEMM dispatches; no new
scratch/stream ABI. Keep N34816/K5120/T256 and no B reuse. Sanitize GQP flags,
retain private sources and compiler dependencies, and check all 32 cores for
code/data limits. Never reuse an output directory or stale compiler project.

Before interpreting the candidate, require byte-exact individual control
high/low outputs against Models/qwen38-27b/prefill-upgate-replay/run. Preserve
slots 7 and 135 and zero inactive outputs. Combine the four candidate products
in FP32, first summing the two input parts per weight part, then weight parts.
The primary reference remains exact effective weights and original FP32 input.

Keep build/control failures explicit. Use the validated v2 build directories.
The standard harness and small projection buffers avoid full-model residency.
Serial NPU execution and unchanged primary gates are required. Projection
improvement alone does not validate propagated FFN/model accuracy; recheck
causal outputs before promotion because up/down errors partly cancel.


## Validated result — 2026-10-20

Both 32-core builds pass resource checks: max text 4784/5568 bytes (control/low),
max buffer end 61448 bytes, configured stacks 4096 bytes, 801 pinned compiler
dependencies each. Toolchain: mlir-aie 1.4.2, Peano
21.0.0.2026080301+c9c5ecb7. Direct aie::to_float of BF16 vectors fails with this
API; widen high parts via accfloat.from_vector then to_vector<float> before
subtracting. Preserve the original failure; v2 builds pin the corrected builder.

Eight native dispatches pass. Every individual control high/low output matches
retained production bytes, candidate slots 7/135 match, inactive columns zero.
Against original FP32 input/exact weights, up maxrel improves
0.0019482680 → 0.0000015091 and gate 0.0002586737 → 0.0000013845. Separate
maxrel and rel_fro must strictly improve; equal results are not improvement.
Host tests: 1522 passed / 47 skipped, 9 new RED-to-GREEN cases.

Evidence: Models/qwen38-27b/prefill-paired-weights/v2/*/build.json and
replay/report.json, captures, logs, validation-evidence.json. Runbook records
binary hashes and exact commands. No production kernels or weights changed.
Next compose the improved up/gate with host activation and original down on
captured causal inputs, retaining exact control and independent layer-output
reference. Check cancellation and broader positions before integration.
Four GEMM products replace two; throughput is unmeasured. Full-model
long-prefix accuracy is still unresolved (558/774 retained logit gates).
