---
name: open-qwen38-fp32-conv-history
description: Build and validate isolated FP32 conv-history storage over FP32 linear entry, using half-tile streams, causal state and a byte-exact BF16-history control.
---

# FP32 conv-history replay

Use after the isolated FP32 linear-entry experiment. Both variants preserve
FP32 entry norm through QKV/Z/alpha/beta. Only the candidate changes conv-history
storage and its MAC input handling. This is a forced-input full-lx replay,
not full-depth propagation or production integration.

## Build

Run from the repository root using native Python 3.14, mlir-aie 1.4.2 and Peano
21.0.0.2026080301+c9c5ecb7. Build/NPU access needs tool escalation in the managed
workspace. The prior linear-entry experiment supplies the verified control.
The builder checks its source, geometry, generated entries and binary hashes.
It automatically generates private recipe-specific kernel translation units;
never rely on global generated C++ files left by another recipe.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
for mode in bf16 fp32; do
  ironvenv/bin/python utilities/conv-history-replay/build.py \
    --kernel-dir Models/qwen38-27b/production-fp32-entry/kernels \
    --control-build Models/qwen38-27b/linear-entry-replay/fp32-build \
    --out "Models/qwen38-27b/conv-history-replay/$mode-build" --mode "$mode"
done
ironvenv/bin/python utilities/attention-entry-replay/resources.py \
  --build Models/qwen38-27b/conv-history-replay/fp32-build \
  --out Models/qwen38-27b/conv-history-replay/build-audit.json
```

Here `bf16`/`fp32` name the **history** dtype, not the entry dtype. Both use
405504 bytes of act scratch. Candidate state grows from 3256320 to 3317760 bytes;
recurrent state starts at 122880 instead of 61440. Model weights are unchanged.
Each original 1024-channel conv tile is processed as two 512-channel halves.
A call consumes four 2-KiB elements (QKV plus three FP32 history rows) and emits
three state elements. Weight tiles remain held across both calls; all eight
value-head records follow the second half. FIFO depths and stack are unchanged.
The host sends strided 3×2048-byte history slices and covers each history byte
exactly once. Main-core recurrence and the post/FFN path are unchanged.

The candidate built in 138.1 seconds. Resource audit passes all 11 cores:
main .text 15248, glue 14400, norm 3600, post 5696 bytes (<16384). Maximum
compiler L1 buffer end is 62336 (<65536), versus 64384 for the control; stack
remains 6144 bytes. Half-tile acquisition permits the allocator to reuse space
without widening the existing FIFOs. The xclbin SHA256 is
`03292dd9ce508f2236251c36c0504b59f04379d121a5411d5773865d9c5848af`.

## Replay and score

The retained incoming residuals come from `head-state-validation/decode`; the
control captures come from the previous `linear-entry-replay/fp32` experiment.
Use one pool/state allocation across layers, resetting between layers only.
Run variants serially on the NPU. No full-model residency is needed.

```bash
mode=bf16 # repeat with fp32
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/conv-history-replay/history_check.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-entry/kernels \
  --fixture Models/qwen38-27b/head-state-validation/decode \
  --pools Models/qwen38-27b/production-full/pools \
  --control Models/qwen38-27b/linear-entry-replay/fp32 \
  --build "Models/qwen38-27b/conv-history-replay/$mode-build" \
  --out "Models/qwen38-27b/conv-history-replay/$mode"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-fp32-harness/run_kernel \
  "Models/qwen38-27b/conv-history-replay/$mode/run.cfg"
```

Repeat the prepare command with `--score` to compare all 48 layers × three
positions. Preparation removes stale output captures. Scoring verifies pinned
inputs, binaries, generated sources and config. The shared DeltaNet checker
accepts an explicit history dtype, requires a bit-exact causal shift (including
signed zero), and preserves the unrounded FP64 primary history/state oracle.
The default BF16 checker behavior and its numerical rows must reproduce the
retained baseline exactly. Head-local diagnostics remain separate and visible.

```bash
ironvenv/bin/python utilities/conv-history-replay/compare.py \
  --root Models/qwen38-27b/conv-history-replay \
  --baseline-score Models/qwen38-27b/linear-entry-replay/fp32/score.json
```

Do not reuse old preparation audits after checker source changes. The control
copies pinned binaries but must be executed and scored afresh against retained
captures. Local acceptance requires 432 byte-identical control captures and
matching oracle/effective-weight rows, common input hashes and complete coverage.
A primary FAIL writes JSON and exits 1. A whole-tensor PASS does not imply every
head passes or authorize promoting an unvalidated full-depth production export.


## Retained result (2026-10-14)

Both variants completed 144 NPU dispatches. BF16-history control reproduced
all 432 retained captures bytewise and every numerical/effective-weight row
from the previous FP32-entry experiment. It still fails one composed state.
FP32 history passes all 144 primary cases under unchanged thresholds.

| Metric | BF16 history worst normalized error | FP32 history worst normalized error |
|---|---:|---:|
| Conv history | 0.0036133307 | 0.0001317673 |
| Composed state | 0.0054339342 | 0.0004810105 |
| Composed output | 0.0044733739 | 0.0014921351 |

The former L57/T1 state failure falls to 0.0000666824. Entry norm/QKV metrics
are unchanged. All Q/K/V/decay/beta fields, 144 conditioned glue/recurrence
checks, 144 bit-exact history shifts and padded states pass. All 6912 head-state
diagnostics pass; **100/6912 head-output diagnostics fail** (control: 95).
Worst absolute error among failed candidate heads is 1.09956607e-7. These
failures remain visible and are not covered by the whole-tensor PASS.

TDD: 14 new tests cover explicit storage, low bits and signed zero, malformed
captures, shared unrounded oracle, half-tile FIFO bounds, complete strided byte
coverage and causal replay configuration. Full host suite: **1224 passed,
47 skipped**. Evidence, logs and hashes live in
`Models/qwen38-27b/conv-history-replay/`. Production binaries are unchanged.
Next integrate both FP32 entry and history into the declared production ABI,
rebuild the complete export and repeat full-depth validation. This isolated
result does not validate serving, prefill or independent all-head acceptance.
