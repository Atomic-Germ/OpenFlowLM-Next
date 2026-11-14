---
name: open-qwen38-paired-prefill-input
description: Build private control and paired-BF16-input hybrid GEMM CLIs, reproduce the full-depth prefill baseline and audit causal prompt/continuation precision without changing production kernels.
---

# Qwen3.8 paired-input prefill experiment

Use to investigate FP32-to-BF16 activation rounding in the host-driven hybrid
GEMM route. This is a private CLI experiment, not the production default. The
validated geometry is H5120, FFN17408, 48 linear value heads, 64 layers, T256.
Read [the runbook](../../../utilities/prefill-paired-input/README.md) for exact
build, reference-copy, serial NPU run and audit commands.

`build.py` copies the standalone engine, preserves its single-dispatch body and
wraps `Core::gemm_run` in the paired candidate. High input uses the original
`tile_x`; low input is `x - bf16_rne(x)`, rounded by the same tiler. Copy high
output before the next dispatch reuses its buffer. Sum only in host storage:
dirty mapped output can overwrite device results during `read_back` CLFLUSH.
Keep per-output-global workspace to preserve returned-pointer lifetimes.
The candidate rejects stage-major because that path bypasses `gemm_run`.

Require at least 35 GiB available RAM and run resident processes serially. The
validated run had about 70 GiB available. Keep the historical reference/captures
in `Models/qwen38-27b/prefill-full/short` intact. Copy references into fresh
control/paired run directories; do not regenerate CPU states from candidate
outputs. The ordinary checker strips ambient OFLM_OPEN flags and selects
non-staged routes explicitly. Each variant runs sequential NPU, hybrid
layer-major and hybrid block-major with one NPU continuation.

Build with GCC 15.2.0, CMake Release, OpenMP and XRT. Local development SDK:
`/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt`;
runtime `LD_LIBRARY_PATH=/opt/xilinx/xrt/lib`. The runbook records CLI hashes;
`build-evidence.json` pins original/staged sources, commands and compiler.
Use fresh build directories; never patch production files for this replay.

Reuse the 41-set/82-hash production-consistent export, build key
`ac29fbcd21859648ff9d59f79e77f018b99daec0888f77a693e7ff18b2237032`.
No new XCLBIN, weight repacking or scratch ABI is needed.

Acceptance on 2026-10-19: control reproduces all 66 historical logits and the
old position-17 hybrid failure. Candidate passes 63/63 independent comparisons;
hybrid min correlation improves 0.9998908204 -> 0.9999488108, final L63
continuation maxrel 0.0018923571 -> 0.0011956505. Thresholds remain >0.9999 and
<0.005. Sequential identity 22/22; hybrid schedule identity 22/22; IDs [17,478].
CTest 4/4 for each build; host suite 1436 passed, 47 skipped; six RED-to-GREEN tests and five additional
audit regressions.
Evidence: `Models/qwen38-27b/prefill-paired-input/comparison.json` and each
variant's `build-evidence.json` / `run/report.json`.

The control report intentionally fails numerically. An execution error is not
that expected failure. The auditor recomputes numerical gates from arrays,
checks hashes and exact control/sequential identities, and fails on incomplete
evidence. Never relax thresholds or substitute matching token IDs for logits.
This demonstrates an input-rounding contribution, not a unique explanation of
all error. Full-depth block boundaries, intermediate residuals, repeated-request
state and the doubled GEMM cost remain to validate before production promotion.
