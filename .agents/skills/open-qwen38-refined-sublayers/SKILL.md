---
name: open-qwen38-refined-sublayers
description: Replay the refined Qwen3.8 candidate's L50/P205, L54/P125 and L60/P220 linear layers with exact native controls and separate propagated versus conditioned DeltaNet/FFN attribution.
---

# Refined candidate causal linear boundaries

Read `utilities/refined-sublayers/README.md` for commands and limitations.
Use after the complete but numerically failing `refined-runtime/native-v1`
audit. Never promote experimental ax arithmetic based on these diagnostics.

1. Source `utilities/env.sh`; use `ironvenv/bin/python` and one OpenBLAS thread.
2. Build the unchanged generic harness with `replay.py build`. Pin compiler
   dependencies and XRT inputs. No new xclbin is needed.
3. Prepare into a fresh directory. Preserve original full-depth and independent
   reference metadata; never reassign its provenance to current source.
4. Run all three layers in separate fresh processes under the device lock.
   Each uses one production-packed pool, unchanged lx binaries and every
   captured candidate input from zero. Never seed captured recurrent state.
5. Require all 553 outputs to reproduce retained native residuals bytewise;
   run the separate audit over 1659 output/activation/state captures before
   interpreting any boundary.
6. Run and separately audit `analyze.py`. Reconstruct original target reference
   outputs exactly and carry independent mixed/FP64 histories. Preserve strict
   per-head gates and both FFN down-partial checks. The post-output slot is
   appended FP32; do not reuse the historical L0 BF16 reader.
7. Keep signed effect vectors and cancellation. Local conditioned correctness
   does not establish full-model correctness. Raw FFN up/gate are not retained.
   Complete numerical FAIL exits 1 and remains distinct from execution failure.

Retained hardware checkpoint: `Models/qwen38-27b/refined-sublayers/harness-v1`
and `native-v2`, 553/553 exact identities and 1659 audited captures. `native-v1`
failed preflight before dispatch and is retained separately. The production
export, installed model, registry and services are unchanged. No performance
acceptance, serving isolation or model-support claim is implied.

`analysis-v1` and its separate audit complete with numerical FAIL and local
PASS. All 553 entry/conv/state and 26,544 strict recurrence head cases pass,
as do selected Z/post/out/FFN/down checks. All three original reference rows
reconstruct exactly. Total errors 0.0117765736 / 0.0066132466 / 0.0620054922
are dominated by propagated current-input drift; local gaps are only
0.0000377089 / 0.0000045350 / 0.0000112520. Next trace upstream current-input
drift rather than modifying these passing target sublayers. Keep the signed
vectors and all three regression controls. Host suite: 2073 passed, 47 skipped;
retained plain and trace CLIs CTest 4/4 each.
