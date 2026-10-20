---
name: open-qwen38-prefill-residual-trace
description: Build a private paired-input Qwen3.8 native CLI with synchronized per-layer residual capture and audit long-prefix drift against byte-exact baseline logits and independent CPU references.
---

# Qwen3.8 native prefill residual tracing

Use to localize the long-prefix precision failure on sequential NPU, hybrid
layer-major and hybrid block-major routes. Read the
[runbook](../../../utilities/prefill-residual-trace/README.md) for exact commands.
This is a private diagnostic build; production kernels and weight packing stay
unchanged. Geometry: H5120, 4 or 64 layers, T256; one continuation after prefill.

Build with `utilities/prefill-residual-trace/build.py` into a new directory. It
reuses the paired-input builder without modifying it and patches only copied
native sources. Sequential capture must disable queue-ahead: global xres can
otherwise be overwritten by the next layer. Read the mapped BO without writing
it. Hybrid captures contain only real rows, after FFN, at absolute positions.
Per-layer files are FP32 `[prompt_tokens+1,5120]`, with no padded rows.

The writer checks contiguous positions, finite values and final coverage and
refuses existing files. The private CLI rejects repeated requests, host/auto or
changing decode routes, stage-major and benchmarks. The native CLI must use
`--pmode none --decode-route npu --prefill-logits --max-tokens 2 --dump-logits`.

Preparation was completed with GCC 15.2.0, CMake Release, OpenMP and XRT headers
from `/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt`.
Build evidence pins original/copied sources, helpers, 585 compiler dependencies,
CMake cache and CLI. CTest 4/4 passes. Retained trace CLI SHA256:
`b3f0bae0a8838b21dd8b8726c7424de5f6cb08826f252fe105fb7b24f62ac1b6`.
All artifacts are under `Models/qwen38-27b/prefill-residual-trace/`.
The 2026-10-20 NPU capture is complete; `native_validated:false` in the build
record is historical preparation status. Current evidence is the separate
`run-257/trace-audit.json`, with `capture_valid:true` and numerical FAIL.

To repeat the capture, run `run.py` with the paired baseline build and a
retained length-257 boundary directory. It requires 35 GiB available for full
residency, verifies provenance, copies independent references into a fresh
output directory and delegates to the existing three-route checker. Do not
regenerate references from native states or overwrite the historical captures.
Runtime uses `LD_LIBRARY_PATH=/opt/xilinx/xrt/lib`; native runs are serial.
The added residual files require about 0.95 GiB for all three length-257 routes.

Read `trace-audit.json` only after `capture_valid:true`. Every prior logit and
alias must match bytewise before interpreting new residuals: synchronization
changes dispatch scheduling, and CPU build/tests cannot prove this identity.
Every layer/position independently requires maxrel <0.005, with exact zero for
zero-reference rows. Missing data or errors invalidate old success. A completed
numeric FAIL is expected while diagnosing the known drift; matching continuation
IDs or final residual alone is insufficient. The first threshold crossing is
not proof of which arithmetic operation caused the error. Do not report trace
CLI timings as production performance or promote paired input on this evidence.


## Measured localization

With 76.87 GiB available at preflight, all three native routes completed 257
prompt positions plus one continuation. All 777 baseline logit files are exact;
all 64 hybrid residual files match bytewise. Residuals pass 46314/49536 gates;
logits pass 558/774. Sequential first exceeds maxrel 0.005 at P38/L60
(0.0090391466); hybrid first exceeds it at P7/L63 (0.0058400461). Indices are
zero-based. Incoming errors are 0.0040749071 / 0.0007421414 respectively.
Worst residual errors: sequential P220/L58 0.5389499720, hybrid P205/L58
0.4325418278. Full host suite 1476 passed / 47 skipped, CTest rerun 4/4.

Use the audit's `localization` section for distinct temporal and depth orderings:
shallowest failing layers are sequential P241/L33 and hybrid P162/L32, not the
first temporal failures. At the first failing logit positions, residual first
crosses at sequential P80/L35 and hybrid P82/L58. The underlying operation is
still unknown. Next capture causal sublayer boundaries at hybrid P7/L63,
sequential P38/L60 and P80/L35, separating incoming error from attention/DeltaNet,
output projection and FFN. Do not infer an arithmetic fix from residual alone.
The length-257 continuation passes; the known length-256 failure remains open.
