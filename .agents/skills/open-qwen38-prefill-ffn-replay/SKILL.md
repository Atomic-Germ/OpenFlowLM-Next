---
name: open-qwen38-prefill-ffn-replay
description: Compose captured paired-weight up/gate with exact host activation and production down GEMM, checking conditioned FFN and independent layer output without model residency.
---

# Composed causal FFN replay

Read the [runbook](../../../utilities/prefill-ffn-replay/README.md) before building
the CPU bridge. Extract current host RMSNorm/activation arithmetic; do not use
NumPy exp as a byte-exact substitute for C++ std::exp. Pin compiler dependencies
and refuse source-anchor drift. Keep production source untouched.

Reuse the validated paired-weight L63/P7 up/gate outputs and original down
kernel. No new XCLBIN. Prove exact norm, control H, down and layer residual;
require zero inactive outputs and byte-exact slot7/135 candidate results.
Require production-pool/source effective weight identity before native execution.
Conditioned FFN and independent final layer-output gates are separate; passing
one cannot mask failure of the other. Do not replace native causal input/state.

A more accurate up/gate may change cancellation with down. Replay the actual
NPU down output rather than freezing its old error. Use fresh directories,
serial hardware processes and pinned baseline/build/capture/reference hashes.
No full-model residency is needed. Isolated FFN acceptance is not propagated
model acceptance.

Measured on 2026-10-20: all eight down dispatches pass execution/identity.
Final output maxrel improves 0.0058400461 → 0.0045267063, crossing the unchanged
0.005 bound. Conditioned FFN maxrel improves 0.0005411159 → 0.0002390673.
Both also pass the additional rel_fro <=0.005 gate. Control norm/H/down/output
are exact; candidate slots 7/135 agree bytewise and inactive columns are zero.
Bridge: GCC 15.2.0, 105 compiler dependencies; 1531 host tests pass, 47 skipped.
Artifacts: Models/qwen38-27b/prefill-ffn-replay/{bridge,run}; reproduction and
bridge hash are in the runbook. Production kernels remain unchanged.

The 2026-10-21 expansion uses `sweep.py` on L60/P38 and L35/P80. Select linear
and attention pool offsets independently; geometry alone cannot select offsets.
Run fresh production and private-control up/gate and compare each high/low
product before interpreting the low-weight candidate. Slots 38/166 and 80/208
must agree at up/gate, H, down and output. Forty dispatches pass exact control,
packing, inactive-column and numerical gates. Candidate up/gate maxrel is at
most 2.241e-6; conditioned FFN improves on both rows. Final-output maxrel grows
slightly to 0.0014800810 / 0.0001962610 while rel_fro improves; both pass fixed
bounds. These hybrid rows already passed; sequential failures are unresolved.

All binaries are reused. Fourteen new RED-to-GREEN tests, 1545 passed and
47 skipped overall. Evidence: Models/qwen38-27b/prefill-ffn-sweep/. Next validate
fresh causal propagation in a private native candidate with exact controls;
never infer model acceptance from selected-row local accuracy.
