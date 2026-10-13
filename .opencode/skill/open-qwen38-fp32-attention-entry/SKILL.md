---
name: open-qwen38-fp32-attention-entry
description: Build and validate an isolated full-ax FP32 attention-entry norm experiment against a byte-exact paired-KV production control, without full-model residency.
---

# FP32 attention entry experiment

Use after `attention_projections.py` finds composed attention-branch failures
although individual projection/head gates pass. This workflow uses the retained
paired-KV export and forced incoming residuals for each layer/position. It does
not change production recipes, manifests, weights or full-model references.

## Design and identity

`utilities/attention-entry-replay/build.py --mode bf16` pins copies of the
retained ax binaries. `--mode fp32` generates a private ax design in its build
directory. Four exact source anchors must match; source drift fails closed.
The design appends 20480 bytes for FP32 entry norm after the original 266240-byte
act buffer. Other offsets remain unchanged. The helper emits two FP32 halves
with `ln_nr32`; Q/gate/K/V preparation consumes five 4-KiB FP32 elements. The
remaining paired attention, output projection and FFN stages reuse production
kernels. The norm worker keeps the existing stack and FIFO topology.

The candidate is not a production export and must not be used with an edited
production manifest. Build metadata pins the production source key, experiment
sources, generated design and both binaries. `entry_check.py` knows the private
appended norm region and calls the same unrounded `score_layer` oracle used by
production validation. Source/weight/input hashes are verified before scoring;
score files are invalidated before each prepare/score attempt.

## Build

Use the native toolchain from `../open-qwen38-production-preflight/SKILL.md`:
Python 3.14, mlir-aie 1.4.2, Peano 21.0.0.2026080301+c9c5ecb7. Build forkserver
and NPU access need sandbox escalation in this environment.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
for mode in bf16 fp32; do
  ironvenv/bin/python utilities/attention-entry-replay/build.py \
    --kernel-dir Models/qwen38-27b/production-paired-attention/kernels \
    --out "Models/qwen38-27b/attention-entry-replay/$mode-build" --mode "$mode"
done
ironvenv/bin/python utilities/attention-entry-replay/resources.py \
  --build Models/qwen38-27b/attention-entry-replay/fp32-build \
  --out Models/qwen38-27b/attention-entry-replay/build-audit.json
```

The audit checks every linked compute core's program size and every declared
compute-buffer end address in `input_with_addresses.mlir`. Basic-sequential
allocation includes the reserved stack in those address ends. Keep .text below
16384 bytes and L1 end at or below 65536 bytes; never lower stack to force a fit.

## Prepare, run and score

For each mode, prepare and run serially on the NPU:

```bash
mode=bf16 # repeat with fp32
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 ironvenv/bin/python \
  utilities/attention-entry-replay/entry_check.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-paired-attention/kernels \
  --fixture Models/qwen38-27b/production-paired-attention/decode \
  --pools Models/qwen38-27b/production-full/pools \
  --build "Models/qwen38-27b/attention-entry-replay/$mode-build" \
  --out "Models/qwen38-27b/attention-entry-replay/$mode"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-fp32-harness/run_kernel \
  "Models/qwen38-27b/attention-entry-replay/$mode/run.cfg"
```

Repeat the preparation command with `--score`. The default covers all 16
full-attention layers and all three fixture positions. One pool, state and act
buffer are reused; state/act are zeroed between layers, while K/V history is
retained across positions. Control activation/residual dumps must reproduce
all retained production bytes. Numeric failure returns exit 1 even for the
expected failing BF16 control. Do not mistake hardware completion for a score.

```bash
ironvenv/bin/python utilities/attention-entry-replay/entry_compare.py \
  --root Models/qwen38-27b/attention-entry-replay \
  --baseline-score Models/qwen38-27b/attention-projections/results.json
```

Comparison checks pinned artifacts, identical common inputs and effective
weights, exact case coverage and BF16 reproduction of the retained oracle
results. Require every FP32 primary norm/projection/output/composed gate to pass
the unchanged normalized max-error bound 0.005. Passing this experiment does
not establish 64-layer propagation, runtime/reset, prefill or model support.
Production integration needs an explicit wider ABI, a complete new export and
new full-depth gates; preserve the retained export's failure marker.

## Verified result — 2026-10-13

Retained export source key `fa34220099d565bfeb514765e3c72665f768a42ee17e9ba37dd7fa3402788405`;
candidate source hash `fcbd7273f296c66ed434ee19da637291286dc6a36408dadef7983d76901b35e5`.
Both variants complete 48 dispatches. All 96 BF16 control activation/residual
buffers reproduce retained production bytes. Control has 10/48 branch failures;
FP32 passes all 48 entry/output/composed cases. Worst branch error falls from
0.0236300737 to 0.0005290482. Worst FP32 norm/Q/gate/K/V errors are
0.0000103846/0.0001849842/0.0002755196/0.0002652475/0.0002784229.
Worst composed residual error is 0.0000867198. Bounds and oracle are unchanged.

Candidate .text max 13760 bytes (main GEMV 9824, norm 3600); 15 compute cores
fit. Max compiler L1 end 59920 bytes with the original 6144-byte stack. Scratch
is 286720 bytes; total logical data BOs 254085120 bytes (~242.31 MiB).
Host suite: 1090 passed, 47 skipped. No production acceptance is inferred:
full integration and new full-depth/runtime gates are still required.
