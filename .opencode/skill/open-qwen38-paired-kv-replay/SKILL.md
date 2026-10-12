---
name: open-qwen38-paired-kv-replay
description: Build and validate isolated Qwen3.8 attention with paired-BF16 K/V and FP32 output, including a byte-exact BF16 control and local memory constraints.
---

# Paired K/V precision experiment

Use `utilities/attention-kv-replay/` to test K/V representation independently
of projection weights or full-model propagation. It supports the Qwen split
geometry NH24/KVH4/HD256/NHL4, VEXP=1, RB=1. Production attention, cache ABI,
model weights and packing are unchanged by this experiment.

The `bf16` control writes zero low planes and calls the production row kernel.
The `pair` variant keeps K/V hi+lo through all four QK products and both PV
value terms, using the same online softmax and FP32 gated finish. Q and K norm,
RoPE, sigmoid and debug capture use existing production/replay functions.
Both variants have the same widened transport. The independent unrounded
FP64 oracle and primary bounds remain absolute max error <0.02 and cosine
>0.9999 per head; conditional diagnostics cannot override them.

## ABI and resource constraints

- Each cached row is `[K_hi, K_lo, V_hi, V_lo]`, four BF16 planes of KVH×HD,
  **8192 bytes** instead of 4096. History is generated on the NPU across all
  positions in a dispatch. The first dummy row is masked; each layer starts
  from zero cache. Keep the earlier `attention-replay` artifacts separate.
- Input FIFO elements remain EA=2048 bytes. Acquire four for a cached row;
  the compiler allocates five input slots. K and V scratch each grow to 4096 B.
- FP32 output FIFO **depth 1** is necessary for this diagnostic design. Depth 2
  failed sequential L1 allocation at **68112 >65536 bytes** per core.
  Depth 1 saves 4096 bytes. Do not reduce the stack to make the design fit.
- Core 0 emits KV and debug on one FIFO, output on another: there are only two
  output DMA channels. Other cores emit output and debug. Debug remains raw
  Q hi/lo, O accumulator and padded softmax max/sum bytes.
- Paired arithmetic uses rolled HD loops to limit program size. Check actual
  linked core ELF `.text` against 16384; isolated fit does not prove full ax fit.
- For three positions the four data BOs total **816 KiB** (input 180, KV 24,
  output 72, debug 540 KiB). No resident model/projection pools are required.

## Build and run

Use the native mlir-aie 1.4.2/Peano environment and harness described in
`../open-qwen38-attention-replay/SKILL.md`. Native compiler forkserver and NPU
dispatch require sandbox escalation on the recorded host. Do not run
source-generating tests while compilation is active.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
for mode in bf16 pair; do
  ironvenv/bin/python utilities/attention-kv-replay/kv_build.py \
    --spec Models/qwen38-27b/production-fp32-front-kernels/spec.json \
    --mode "$mode" --tokens 3 \
    --out "Models/qwen38-27b/attention-kv-replay/$mode-build"
  ironvenv/bin/python utilities/core_sizes.py \
    "Models/qwen38-27b/attention-kv-replay/$mode-build"
  OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/attention-kv-replay/kv_check.py \
    --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
    --kernel-dir Models/qwen38-27b/production-fp32-front-kernels \
    --fixture Models/qwen38-27b/production-fp32-front-multitoken \
    --build "Models/qwen38-27b/attention-kv-replay/$mode-build" \
    --out "Models/qwen38-27b/attention-kv-replay/$mode" --mode "$mode"
  HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
    /tmp/qwen38-fp32-harness/run_kernel \
    "Models/qwen38-27b/attention-kv-replay/$mode/run.cfg"
done
```

Repeat each preparation command with `--score`. The control is expected to
return exit 1 for its numerical failures; hardware exit 0 alone is insufficient.
Preparation and scoring pin source, geometry, mode, binaries, constants and
inputs; scores are invalidated before validation. The control rejects nonzero
low planes, and both modes reject non-finite or incorrectly sized captures.

The fixture/export key and the replay source key are distinct: upstream recipe
changes may move the latter without changing retained projections. Verify the
fixture against its original export and the replay against current sources,
geometry and binary hashes. Both keys are recorded in `audit.json`; never
relabel the old export as a new production build.

Verify the controlled experiment against the retained FP32-output replay:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/attention-kv-replay/kv_compare.py \
  --original Models/qwen38-27b/attention-replay/fp32 \
  --control Models/qwen38-27b/attention-kv-replay/bf16 \
  --pair Models/qwen38-27b/attention-kv-replay/pair \
  --out Models/qwen38-27b/attention-kv-replay/results.json
```

This verifies input/capture hashes, coverage, identical inputs and Q, exact
control output/debug/KV reproduction, and identical high planes across modes.
It requires the paired variant's independent primary gate to pass.

For a machine-readable program/L1 audit, run:

```bash
ironvenv/bin/python utilities/attention-kv-replay/kv_resources.py \
  --root Models/qwen38-27b/attention-kv-replay
```

## Recorded result — 2026-10-12

Both modes completed 16 dispatches × three positions × 24 heads. The widened
control reproduces previous FP32 output, debug and reconstructed KV bytewise,
including **21 FAIL**, worst absolute **0.1974321142**, L59/T2/H9. All inputs,
Q and high K/V planes agree across variants. Paired K/V has **1152 PASS**,
worst absolute **0.0009283289** at the same head, minimum cosine
**0.9999999990516776**. Primary thresholds and the independent oracle are unchanged.

Across layers, K boundary error falls from **0.0613304117** to **0.0002521316**;
V from **0.1993179321** to **0.0001831055**. Conditional softmax/PV and output
checks also pass. Program maxima: control **15184**, pair **14992** bytes.
Actual allocated L1 ends at **64016 /65536 bytes**, including the 6144-byte
stack, leaving 1520 bytes. This is an instrumented six-core replay only.

Evidence: ignored `Models/qwen38-27b/attention-kv-replay/`, build/audit/score
JSON, `results.json`, `build-audit.json`, captures and logs. The initial L1
failure logs are retained. Host suite: **1038 passed, 47 skipped**, including
15 new tests (layout/precision and independent replay/fixture identities).

Production attention remains uncorrected. Next assess full ax program/L1,
cache streaming and the FP32 output-to-projection contract before integration
and a new complete export/full-depth regression. No new full-model run was
performed in this experiment.
