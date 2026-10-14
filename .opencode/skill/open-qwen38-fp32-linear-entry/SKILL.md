---
name: open-qwen38-fp32-linear-entry
description: Build and validate isolated full-lx FP32 norm-to-QKV/Z/alpha/beta with a byte-exact production control and causal conv/recurrent history.
---

# FP32 linear-entry experiment

Use after `deltanet_inputs.py` exposes composed-state failures despite passing
captured-input glue and recurrence. This private full-lx design appends FP32
entry scratch; it does not alter the production manifest, kernel export or
weight pools. Every layer uses retained incoming residuals, with causal conv
and recurrent state across positions. It does not test corrected full-depth
propagation, runtime/reset, serving or prefill.

## Build and resource audit

Use native Python 3.14, mlir-aie 1.4.2 and Peano
21.0.0.2026080301+c9c5ecb7, as in production-preflight. Build forkserver and NPU
access require execution-tool escalation in this environment. No full model
needs to reside in memory; file hashes stream through bounded host buffers.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
for mode in bf16 fp32; do
  ironvenv/bin/python utilities/linear-entry-replay/build.py \
    --kernel-dir Models/qwen38-27b/production-fp32-entry/kernels \
    --out "Models/qwen38-27b/linear-entry-replay/$mode-build" --mode "$mode"
done
ironvenv/bin/python utilities/attention-entry-replay/resources.py \
  --build Models/qwen38-27b/linear-entry-replay/fp32-build \
  --out Models/qwen38-27b/linear-entry-replay/build-audit.json
```

The BF16 control copies verified production lx binaries. The candidate changes
six exact source anchors and refuses source drift. Entry norm uses the shared
FP32 worker; QKV/Z use direct FP32 table preparation. Alpha/beta consume five
1024-float blocks in the existing 4-KiB scratch slot, with 32 unchanged weight
tiles per block. BF16 weights multiply both high and low parts of FP32 inputs.
Conv history remains BF16; conv, recurrence, post and FFN arithmetic are unchanged.

The builder automatically generates recipe-specific C++ translation units into
`fp32-build/generated`, uses the shared layer headers explicitly, and verifies
exact source coverage/content before and after compilation and during replay.
Their hashes are recorded in `build.json`. Do not compile against leftover
`layer_x/*.cc`: a stale `GEMV_PER_CALL=2` instead of 1 can produce non-finite
outputs, and stale dense prep can silently reintroduce BF16 rounding. There
is no manual generation prerequisite or change to production-generated files.

Roll the five AB blocks and their tile loops with the recipe's no-unroll loop.
The first attempt emitted five copies and overflowed glue program memory:
17344 >16384 bytes. The rolled version preserves all 160 weight tiles per
projection and resets accumulators only at block 0/tile 0. It fits: glue text
14176, main cores 15248, norm 3600, post 5696 bytes. All 11 cores fit; maximum
compiler L1 end 64384 with unchanged 6144-byte stack. Never shrink stack to fit.
Act scratch grows from 385024 to 405504 bytes; state stays 3256320 bytes.

## Prepare, run and score

For each mode, run the following serially on the NPU. The default covers all
48 linear layers × three positions, reusing one pool and state buffer. Reset
state between layers only. `--layers` can select a covered subset for debugging;
it does not establish whole-fixture acceptance.

```bash
mode=bf16 # repeat for fp32
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/linear-entry-replay/linear_check.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-entry/kernels \
  --fixture Models/qwen38-27b/head-state-validation/decode \
  --pools Models/qwen38-27b/production-full/pools \
  --build "Models/qwen38-27b/linear-entry-replay/$mode-build" \
  --out "Models/qwen38-27b/linear-entry-replay/$mode"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-fp32-harness/run_kernel \
  "Models/qwen38-27b/linear-entry-replay/$mode/run.cfg"
```

Repeat the preparation command with `--score`. Preparation removes stale
outputs; scoring verifies source, config, model, pool and input hashes. Each
position dumps complete act/residual/state buffers; the control must reproduce
all three production captures bytewise. `deltanet_inputs.score_sequence` is the
shared unrounded oracle. Record preparation, state/output and conditioned
glue/recurrence bounds are unchanged; per-head diagnostics remain visible.
Numerical failure writes complete `score.json` and returns exit status 1.

```bash
ironvenv/bin/python utilities/linear-entry-replay/compare.py \
  --root Models/qwen38-27b/linear-entry-replay \
  --baseline-score Models/qwen38-27b/deltanet-input-validation/input-results.json
```

Comparison requires identical common input/effective-weight hashes, complete
case coverage, and control oracle equality with retained production results.
Keep any remaining conv-history/state or head-output failures visible. Even a
local PASS would require production ABI integration, a complete rebuilt export
and new full-depth validation before claiming corrected model propagation.


## Retained result (2026-10-14)

Both variants completed 144 NPU dispatches (48 layers × three positions). The
BF16 control reproduced all 432 captures bytewise and every oracle row from
`deltanet-input-validation/input-results.json`. The FP32 candidate passed
143/144 cases versus 121/144 for BF16, with unchanged acceptance bounds.

| Primary metric | BF16 worst normalized error | FP32 worst normalized error |
|---|---:|---:|
| Entry norm | 0.0035667850 | 0.0000161226 |
| QKV | 0.0032124482 | 0.0001637651 |
| Composed state | 0.0100545744 | 0.0054339342 |
| Composed output | 0.0317001700 | 0.0044733739 |

All FP32 Q/K/V/decay/beta record fields and 144 outputs pass. One composed state
fails at L57/T1 against 0.005. Conditioned glue and whole-tensor recurrence pass
144/144; all 6912 head-state diagnostics pass, but 95/6912 head-output diagnostics
fail (BF16: 99). Worst absolute error among failed FP32 heads is 1.18928361e-7.
Do not present the candidate as complete numerical acceptance.

At L57/T1 the CPU history-rounding-only diagnostic predicts a 0.0054568534
state deviation from the primary reference and matches the candidate state
within 0.0002856729. This supports testing higher-precision conv-history storage
next; it is not proof that such a kernel correction passes. Across all cases
the same diagnostic still has one state mismatch, so retain its full results.

Evidence: `Models/qwen38-27b/linear-entry-replay/` contains the two build records,
resource audit, configs, input/capture hashes, score JSON, comparison and logs.
The candidate xclbin SHA256 is
`54e996fa3209f2fcd280edea09dbef95e7a9753d91246f9d95396e0a15382d69`.
TDD includes source generation, stale-file refusal, private header lookup,
scratch ABI, norm precision and causal/rolled schedules: **12 new tests**;
full host suite **1210 passed, 47 skipped**. Production binaries are unchanged.
