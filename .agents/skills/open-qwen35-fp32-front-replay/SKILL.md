---
name: open-qwen35-fp32-front-replay
description: Reproduce Qwen3.8 layer-11 norm-input rounding on the NPU and validate isolated FP32 RMSNorm/up/gate/SiLU with separate projection captures.
---

# FP32 FFN front experiment

Use after `ffn_front.py` identifies L11/T0 activation maxrel 0.0071636206.
This is an isolated FFN-front design from retained pre-FFN residuals. It does
not change the production lx/ax layout or validate full corrected propagation.
The probe runs one norm helper and one GEMV core. Qwen3.5-family Q4 FFN only,
hidden divisible by 1024. Weights use the recipe's unchanged production pools.

The control and corrected builds both run the existing `ln_nr32` FP32 norm.
`FRONT_FP32=0` rounds its input to BF16 during legacy table preparation;
`FRONT_FP32=1` uses direct FP32-to-INT16 preparation. They use identical norm,
up/gate weights, SiLU and inputs. Separate up, gate and h outputs prevent
projection errors hiding behind their product. This is not a whole-layer replay.

## Build

Use the production-preflight skill for model provenance and native XRT setup.
Here Python 3.14 / mlir-aie 1.4.2 and Peano 21.0.0.2026080301+c9c5ecb7 are
installed. Native builds require a local forkserver socket outside the sandbox.

```bash
export OPEN_KERNELS_SPEC="$PWD/Models/qwen38-27b/production-fp32-kernels/spec.json"
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
FRONT_FP32=0 ironvenv/bin/python open_kernels/build_design.py \
  utilities/ffn-front-replay/probe.py Models/qwen38-27b/ffn-front-replay/baseline-build
FRONT_FP32=1 ironvenv/bin/python open_kernels/build_design.py \
  utilities/ffn-front-replay/probe.py Models/qwen38-27b/ffn-front-replay/fp32-build
ironvenv/bin/python utilities/core_sizes.py Models/qwen38-27b/ffn-front-replay/fp32-build
```

This design has its own source files; it does not depend on generated layer_x
translation units. External entries live in separate `.cc` files because aiecc
compiles one object per ExternalFunction: sharing a multi-entry source produces
duplicate linker symbols. `Worker` placement uses `tile=`, not `placement=`.
Source/spec/mode hashes prevent control and corrected builds sharing a JIT key.

## Prepare, run, score

Use a separate output directory for each build. The default layer is 11 and all
fixture positions are included. The example below prepares the FP32 variant;
replace `fp32` with `baseline` in both paths for the control.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/ffn-front-replay/check.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-kernels \
  --fixture Models/qwen38-27b/production-fp32-multitoken \
  --pools Models/qwen38-27b/production-full/pools \
  --out Models/qwen38-27b/ffn-front-replay/fp32 \
  --build Models/qwen38-27b/ffn-front-replay/fp32-build
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-fp32-harness/run_kernel Models/qwen38-27b/ffn-front-replay/fp32/run.cfg
```

Repeat the preparation command with `--score`. It verifies the pinned kernel,
configuration, model, pool, source captures and generated input hashes before
checking outputs. `score.json` is invalidated before each preparation/score;
failures return nonzero. Model hashing streams the file, not a whole-file read.
One pool and one layer's matrices are needed, not the resident 64-layer NPU model.
The host harness in `/tmp` may need rebuilding; see the production-preflight skill.

`norm_tN.bin` is FP32[hidden]; `front_tN.bin` contains bands of
[up[64], gate[64], h[64]]. Do not read it as three contiguous whole vectors.
Primary up/gate/h comparisons use FP64 RMSNorm of the original captured residual,
without BF16 rounding. Each has maxrel <0.005, including the independent norm
and combined-activation checks. Diagnostic rounding cannot override failure.

Production integration still needs a wider norm-to-up/gate stream, explicit
buffer layout/manifest handling and whole-layer program-memory validation.
A small probe fitting program memory does not prove lx/ax fits. After integration,
rerun local gates and the full chain; the seven existing residual failures remain
unresolved until that run passes. Do not promote the experimental xclbins.

## Verified result — 2026-10-11

Layer 11, three retained positions, three dispatches per build. Control h is
byte-identical to the original full-layer capture at every position, including
the failing T0 maxrel 0.0071636206. Corrected T0 h maxrel is 0.0001517240.
All three FP32 norm/up/gate/h comparisons pass the unchanged 0.005 threshold.
Worst errors: norm 0.0000063849, up 0.0000937135, gate 0.0000796810,
h 0.0001517240. Both builds produce byte-identical FP32 norm buffers and use
identical residual/weight inputs; only table preparation changes.

Corrected probe program .text: GEMV core 6304 B, norm helper 2944 B (<16384).
Artifacts live in ignored `Models/qwen38-27b/ffn-front-replay/`, including
`baseline/score.json` (FAIL), `fp32/score.json` (PASS), pinned inputs, build and
hardware logs, core sizes and `results.json`. These results establish the local
fix only. The production export and its full-chain FAIL status remain unchanged.
