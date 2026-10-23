---
name: open-qwen38-prefill-attention-replay
description: Build private attention entry control/paired-weight kernels and compose causal paired KV, output and FFN with exact native controls.
---

# Causal full-attention prefill replay

Read [the runbook](../../../utilities/prefill-attention-replay/README.md).
Build private N14336/K5120/T256 Q4_1 control/low kernels, no B reuse; reuse the
verified output, up/gate and down kernels. Register this workflow before building.
Sanitize GQP flags, retain failures and pin source/generated/compiler dependency,
32-core resource and binary evidence. Build fresh production attention and
extracted RMSNorm/activation CPU bridges with contraction disabled.

Use the audited complete 20-token L63 prefix and selected P1. Validate original
attribution, capture/baseline/model provenance, all packed/source identities and
fresh exact native controls before interpreting candidates. Check paired KV and
attention block/single-token identity, zero inactive columns and offsets 0/128.
Keep original per-head entry gates and independent host/projection/conditioned
FFN/final-output gates distinct. Do not substitute ideal incoming residual or KV.

Run serially under the shared device lock with the verified harness; model
residency is unnecessary. Recompute audits from raw outputs without NPU calls.
Local suffix acceptance does not establish fresh model propagation, long-prefix
accuracy, production integration or throughput. Preserve numerical failures.

## Validated checkpoint — 2026-10-23

Artifacts: `Models/qwen38-27b/prefill-attention-replay/`, final `prepared-v2`
and `native-v2`. All 64 dispatches, exact native/individual controls, paired
KV, host partition identity, inactive columns and candidate slot identities pass.
2240 entry heads strictly improve, worst maxrel 4.8850442e-6; 1280 host/cache
and 46 remaining projection gates pass. Independent L63/P1 output maxrel
0.0096485275 → 0.0040805495, rel_fro 0.0017167913: both bounds pass.
Conditioned FFN maxrel 2.5746182e-5; CPU re-audit recomputes all results.

Builds: 801 dependencies each, 32-core text 4784/5568 bytes, L1 buffer end
61448 bytes, stack 4096 bytes. Fresh GCC 15.2 bridges: 342 attention / 105 FFN
dependencies. Full hashes and commands are in the runbook/build records.
Fourteen RED-to-GREEN tests; 1723 host tests pass, 47 skip.

Native output stored by the attribution helper is an FP64 container holding
exact FP32 values. The first run correctly rejected dtype identity after 28
calls. `native_vector` now requires lossless FP32 conversion and finite values;
retain v1 failure/snapshot and use fresh v2 outputs. Never weaken byte identity.

Next extend the private combined CLI to attention entry/out and FFN down, then
run fresh full-depth propagation with independent gates and repeat/reset.
The retained full-depth 4000/4032 residual result remains FAIL. Production
sources and kernels are unchanged; long-prefix/model acceptance stays open.
