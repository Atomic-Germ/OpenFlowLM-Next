---
name: open-qwen38-prefill-sublayer-trace
description: Build a private native Qwen3.8 prefill sublayer tracer, preserve causal history and byte-exact baseline outputs, and separate propagated input error from conditioned FFN precision.
---

# Qwen3.8 selected causal sublayer capture

Use for residual failures at hybrid L63/P7, sequential L60/P38 and L35/P80
(zero-based). Read the [runbook](../../../utilities/prefill-sublayer-trace/README.md)
for build, resident run and CPU audit commands. This diagnostic extends the
per-layer tracer; it does not change arithmetic, kernels or packing.

The private builder reuses the residual/paired builders unchanged and patches
copied sources. Its capture header is pinned in compiler dependencies. Reject
an existing output directory before any evidence mutation. GCC 15.2.0,
CMake Release/OpenMP and the local XRT SDK produced CLI SHA256
`dee5a361dcbf5bba6fc8b2e29e182b616652679c05186291d2b88a7091697d7d`.
Artifacts are in `Models/qwen38-27b/prefill-sublayer-trace/`.
Use `verified/build-evidence.json`: its CLI is byte-identical to the first
`instrumented` build used for native capture, but pins the corrected builder
that safely refuses existing directories. Both builds pass CTest 4/4.

Require 35 GiB available before a full resident run, serial NPU processes,
`LD_LIBRARY_PATH=/opt/xilinx/xrt/lib`, and a fresh capture directory. The ordinary
prefill checker runs all three routes through 257 prompt positions and one NPU
continuation. The native engine retains its own causal history; never replace
that history with the CPU oracle's state. No throughput claim from tracing.

Sequential capture reads the selected layer's entire act scratch after its
synchronized dispatch. Hybrid captures real rows only: input, pre-FFN residual,
norm, up/gate, activation and down. Layer-major absolute position is pos_+row0
for linear layers, pos0 for full layers. Block-major row0 is zero. Preserve
this distinction at the 256-token boundary. The FFN helper uses the enclosing
block's start and real-row count; padding must never select a target.

Before interpreting sublayers, the checker requires 777 exact baseline logits,
192 exact residual files, exactly 39 selected captures and byte-identical hybrid
sublayers. Check actual input against previous residual and reconstruct the
output from captured pre-FFN residual/down within FP32 summation error bounds.
Scratch layout and norm dtype must agree with the production manifest.

The primary causal CPU replay uses the retained independent input prefix and
its own state, reproducing the selected stored CPU output bytewise. Another
replay uses native inputs and ideal CPU state; it is diagnostic, not actual NPU
state and not a replacement reference. Stream one layer's weights at a time.

Conditioned FFN checks use actual pre-FFN residual and unrounded FP64 norm:
norm, activation, down from captured H, and composed FFN; hybrid also has
separate up/gate and SiLU gates. Sequential scratch lacks separate up/gate, so
do not claim those measurements. All gates remain maxrel <0.005. Separate
capture validity, conditioned FFN acceptance and global model acceptance.
Passing conditioned arithmetic cannot by itself identify a faulty earlier op.


## Validated results

2026-10-20: 777/777 baseline logits, 192/192 residual files, 39 selected captures
and 18/18 hybrid capture pairs pass identity/coverage. Three independently
reconstructed causal CPU outputs are exact. All 54 conditioned FFN gates pass
across nine cases; worst 0.0024681085 (hybrid front H at L35/P80). Host suite:
1484 passed / 47 skipped, with eight new tests (seven observed RED); CTest 4/4.

Sequential pre-FFN already fails at L60/P38 (0.0093615690) and L35/P80
(0.0084320701). Ideal FFN on actual pre-FFN input reproduces output error almost
exactly (local output errors 8.1696e-6 / 2.7012e-6). Hybrid L63/P7 pre-FFN error
is 0.0008105224; ideal FFN gives output error 0.0046310385, while native gives
0.0058400461. Native vs ideal output error is 0.0031273746. Different reference
scales mean these scalar errors cannot be added. Local hybrid FFN arithmetic
contributes to this crossing, without proving a single faulty operation.
Next isolate up/gate, activation and down contributions at L63/P7; retain the
separate sequential upstream-history investigation. Full-model accuracy remains
FAIL. The initial audit's stale builder-source refusal is preserved, then the
separate audit uses the verified byte-identical binary; do not erase that history.
