---
name: open-down-block-trace
description: Trace real Q4 down blocks, local/persistent compensation and activation sums to isolate errors before segment accumulation.
---

# Real Q4 block trace

Read [the report](../../../specs/open-engine/plans/qwen38-down-block-trace.md).
Prerequisite: the validated `open-down-segment-trace` replay. Use fresh output
directories and retain original fixtures, model weights and acceptance bounds.
No new full-model acceptance is implied by a diagnostic passing.

Use ironvenv Python3.14 / mlir-aie1.4.2 / llvm-aie21 and XRT. Build sequentially.
This probe owns its .cpp files and does not regenerate the shared layer_x TUs.

```bash
base=open_kernels/designs/wide_deltanet/build_down_blocks/sums
OFLM_KEEP_FAILED=1 PATH=/opt/xilinx/xrt/bin:$PATH LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/layer_x/down_block_probe.py "$base"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-down-blocks.py prepare \
  --source open_kernels/designs/wide_deltanet/build_down_trace/retained/layer0 \
  --kernel "$base" --out "$base/channel4872" --segment 3 --channel 4872
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/channel4872/blocks.cfg"
ironvenv/bin/python utilities/diagnose-wide-down-blocks.py compare --out "$base/channel4872"
```

Repeat preparation/run/compare for channel392/segment1 and channel1769/segment3,
each in a distinct directory. Only full K4096 segments 0..3 are supported by
this diagnostic; it does not accept the K1024 tail. Input weights are bytewise
copies of the selected original 32-row pool chunks. Source fixture hashes,
activation guards and full-width trace reproduction are checked before creating
the output directory. Real/zero/real runs test deterministic reset.

The optional GEMV header hook stores four even/odd-permuted planes per block:
local high, local low, accumulated high, accumulated low. Each 1120-float tile
record also contains interface high/low and the activation sum's three BF16
components. See `wide_down_blocks.py` for decoding; final interface high uses
logical lanes, unlike all earlier interface highs. Never apply the same lane
permutation blindly to every field.

Recorded: local errors at block53 K13984..14015 for4872/1769 and block107
K7520..7551 for392. Both final components match the preceding segment trace
exactly. Persistent block reduction and all captured activation sums are exact.
The proportional errors therefore do NOT prove a block-sum defect. Next trace
the nine product components and their local compensation, plus integer dot
scaling, before changing production arithmetic.

The retained probe is6288 B .text. A rebuild of corrected down with this macro
disabled passes66/66 synthetic checks and matches all11 earlier output arenas,
traces and instruction bytes. Full CPU suite:881 passed,47 skipped. Reproduce
that regression using the `open-down-segment-trace` build flags with a fresh
`--out`; no `GEMV_Q4_BLOCK_TRACE` is set by that builder.

`passed` means faithful diagnostic reproduction. The original three down FP32
differences and four full-model slice failures remain; all neural computation
here is on NPU, with independent FP64 references on host for measurement only.
