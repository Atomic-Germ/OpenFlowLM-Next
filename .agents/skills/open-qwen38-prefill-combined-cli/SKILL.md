---
name: open-qwen38-prefill-combined-cli
description: Build fresh paired-input, weighted-FFN and combined linear QKV/Z/out prefill CLIs with role guards and pinned kernel provenance, without model residency.
---

# Combined private prefill preparation

Read [the runbook](../../../utilities/prefill-combined-cli/README.md).
Build fresh directories with utilities/prefill-combined-cli/build.py. Reuse
validated low kernels from prefill-paired-weights/v2/low,
prefill-entry-replay/builds/low and prefill-out-replay/builds/low. Verify all
kernel records and exact geometry before compiling; do not rewrite historical
hashes. No new XCLBIN is required.

Build paired, weighted and combined variants. Preserve the original single
GEMM body and the exact existing control wrappers. Combined corrects only
linear entry/out plus dense FFN up/gate. Full-attention out has the same shape
as linear out, so match kind, role and buffer arguments explicitly. Use
separate low-weight contexts and paired workspaces, and sum in host-owned
FP32 memory. Refuse stage-major and incompatible split/geometry.

Record source/staged hashes, seven compiler depfiles, CLI/cache identity,
compiler version and native CTest. Existing outputs must never be overwritten.
Build completion is not a numerical PASS: native_validated and model_acceptance
stay false. Do not launch resident runs when the user requested preparation
only. Future NPU runs require at least 35 GiB available RAM, exclusive device
access, both fresh controls matching their retained baselines, independent
full-depth residual/logit gates and repeated-request/reset validation. The
single-request tracer must be extended before claiming the latter.

Validated preparation 2026-10-23: v1 paired/weighted/combined CTest 4/4 each;
589/590/591 CLI compiler dependencies. Six RED-to-GREEN tests, full host suite
1672 passed / 47 skipped. prepared-v1 verifies 652 inputs, 192 historical
weighted residuals and exact current model-derived manifest, then retains 149
independent arrays plus IDs/metadata unchanged. Use prepare.py as documented
in the runbook; it hashes model weights incrementally, without engine residency.
All native accuracy and repeated-request/reset gates remain pending. CLI hashes
and commands are in the runbook; logs and records under
Models/qwen38-27b/prefill-combined-cli/.
