---
name: open-qwen38-prefill-down-replay
description: Build isolated paired-weight FFN down kernels and validate combined L63/P1, L42/P17 and L52/P17 inputs with exact native controls and composed residual gates.
---

# Combined FFN down replay

Read [the runbook](../../../utilities/prefill-down-replay/README.md).
Build private control/low copies at N5120/K17408/T256, Q4_1, no B reuse.
Use the established accfloat low-weight conversion. Pin sources, generated
files, compiler dependencies and binaries; check all 32 core text/L1 budgets.
Register this skill before building. Never modify production or prior sources.

Verify current combined capture/attribution provenance, packed/source weights
and original references. Place each H row in P and P+128, other columns zero.
Run production, control and low kernels with both input halves, serially with
exclusive device access and the shared lock. Require individual production /
control byte identity and their sum equal to the native captured down, then
exact reconstructed native residual. Keep control failures explicit.

Combine four products in the same FP32 order as the weighted CLI. Require
candidate placement identity, zero inactive columns, original projection gates
and strictly improved maxrel/rel_fro. Preserve independent composed residual
failures even when projections pass. No full-model residency or promotion;
full-attention and full-depth propagation remain follow-up gates.

## Validated checkpoint (2026-10-23)

Use prepared-v2/native-v2 under Models/qwen38-27b/prefill-down-replay.
All 36 final dispatches complete: individual controls, captured native down,
reconstructed native residual and two placements byte-exact; inactive columns
zero. All six projection gates improve both errors. Down maxrel at L63/L42/L52
is 3.8361e-6 / 2.0427e-6 / 3.4903e-6. Composed residuals still fail:
0.0070247892 / 0.0052110173 / 0.0123485917. L52 worsens from 0.0121972127;
local error partly cancelled incoming drift. Independent audit agrees, and
model_acceptance stays false. Exit 1 for run/audit is expected residual FAIL.

Both 32-core builds pass budgets: text 4784/5568 bytes, L1 buffer end 61448,
stack4096, 801 compiler dependencies each. The existing out-replay harness
verifies 326 dependencies. Exact commands and binary hashes are in the runbook.
The v1 preparation wrote FP32 bytes to an output-reference.f64 file; its 36
finished dispatches were not accepted because scoring rejected the extent.
Retain its failed report/snapshot. The explicit FP64 writer is regression-tested;
fresh v2 preparation/replay and CPU re-audit are the accepted evidence.
Ten RED-to-GREEN tests; suite 1709 passed / 47 skipped.

Next replay full-attention N14336/K5120 fused Q/K/V/gate and N5120/K6144 out
on the causal L63 prefix. Require exact native controls, two placements and
per-head gates before composing with this down correction and expanding private
policy. No production promotion from these isolated results.
