---
name: open-qwen38-prefill-out-replay
description: Build private paired-weight output kernels and replay retained Qwen3.8 gated prefixes after merge with historical fixture provenance, fresh harness controls and independent residual gates.
---

# Paired-weight output projection

Read [the runbook](../../../utilities/prefill-out-replay/README.md). Build private
N5120/K6144/T256 Q4_1 control/low kernels with the validated dequant remainder
helper. Sanitize GQP flags, refuse existing directories, pin generated/compiler
sources and all binaries, audit 32 core text/L1 budgets. Rebuild standard harness
against current source after merge. Register this workflow before compiling.

Retained prefixes come from revision aef4a93 and must not be relabeled current.
Verify all artifacts bytewise; a changed tracked source is valid historical
input only if its expected hash matches that revision and current bytes match
HEAD. No fallback for artifacts or external deps. Record both identities and
validate referenced build/capture evidence recursively. New builds always require
current-source verification. Never rewrite old hashes to make validation pass.

Replay original/corrected-entry gated prefixes at L34/P17, L37/P7 and L53/P17,
20 rows each at two placements. Require exact production individual products
against retained NPU dumps, exact new control products and zero inactive columns.
Candidate combines FP32 (WhXh+WhXl)+(WlXh+WlXl), placement identity mandatory.
Run 72 serial dispatches without full-model residency. Retain independent FP64
projection references and model residual gates separately. Local precision PASS
cannot conceal inherited drift or failing model residuals. No production
promotion, FFN/logit correction or throughput claim.

## Measured checkpoint — 2026-10-22

Artifacts `Models/qwen38-27b/prefill-out-replay/`: complete `prepared-v3/` and
`run/`; earlier preparation failures retained. History audit: 2075 files,
25 records, 23 changed tracked sources tied to old and current revisions.
All 72 dispatches and exact controls/placements pass. 240/240 projection gates
strictly improve; worst candidate maxrel 1.1842201e-5, rel_fro 4.8705972e-6.
Combined entry/out residual L34/P17 0.0006087006, L37/P7 0.0008481152,
L53/P17 0.0066072119: L53 remains FAIL. Exit 1 is expected for this complete
numerical measurement. No full-model propagation or production promotion.

Both builds: 32-core budgets pass, max text 4784/5568 bytes, L1 buffer end
61448 bytes, 801 dependencies each. Fresh harness: 326 dependencies. Hashes
and commands are recorded in the runbook. Fourteen new RED-to-GREEN tests;
full suite 1665 passed / 47 skipped. Next build a private combined entry/out/
up-gate CLI, require a fresh post-merge control, and measure cross-layer drift.
