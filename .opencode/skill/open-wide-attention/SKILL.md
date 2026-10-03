---
name: open-wide-attention
description: Build and validate the production ax attention worker at Q24/KV4/HD256/ROT64, with head-local comparisons, poisoned future rows, device-carried KV state and Q16/KV4 regression.
---

# Wide gated attention primitive

This is an isolated hardware gate for the exact Qwen3.8-27B attention tuple.
The production C++ kernels and `ax._attn` body are reused, with six cores of
four query heads, VEXP=1 and RB=1. It does not validate a full attention layer,
projection packing or the runtime instruction patcher.

From the repository root, using the installed ironvenv/Peano and open harness:

```bash
export XILINX_XRT=/opt/xilinx/xrt
export PATH=/opt/xilinx/xrt/bin:$PATH
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}
attn_build=open_kernels/designs/attn/build_wide_257
ATTN_PROBE_ROWS=257 ATTN_PROBE_FIXTURE=config_qwen38_27b.json \
  ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/attn/wide_probe.py "$attn_build"
ironvenv/bin/python utilities/test-wide-attention.py prepare \
  --rows 257 --fixture config_qwen38_27b.json --build-dir "$attn_build"
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$attn_build/attention.cfg"
ironvenv/bin/python utilities/test-wide-attention.py compare --build-dir "$attn_build"
ironvenv/bin/python -m pytest specs/open-engine/tests/test_wide_attention.py -q
```

For the longer gate use rows2048 and a distinct `build_wide_2048` directory.
For regression use `config_qwen35_9b.json`, rows257, and
`build_regression_16_4`. Match build environment and fixture arguments.
Clear inherited `ATTN_FAST`/`ATTN_RB` probe overrides to use the normal recipe.
The diagnostic accepts VEXP/RB1 with one complete output element per core.
The legacy `attn.py` is hardcoded Q16/KV2 and is not this test's builder.

`probe_support.py` extracts the actual `_attn` function from `layer_x/ax.py`
through AST. Do not copy its math into a new kernel. Its source hash includes
ax, helper, C++/headers, geometry and streamed row count. Probe compute tiles
are(c,2), rather than the full layer's(2+c,3); whole-layer placement remains
a separate measurement.

The six BOs are meta, q|gate, k|v, interleaved KV window, new KV row, gated
output. nf is fixed by the diagnostic build; pos changes per call in meta.
All future rows are NaN to test masking. This exercises the production row
consumer but does not validate runtime-patched DMA lengths/offsets.

Every build tests eight sequential tokens. The harness copies each **device**
new KV row into the next cache slot; no reference values are sent back to the
device. The oracle separately evolves its ideal bf16 cache. Compare the final
device cache to the concatenation of device row dumps, retaining future NaNs
and the canary. A repeat-first case checks reset of private state.

Keep the inherited attention gates: K/V max-relative<1e-2, output<2e-2,
cosine>0.9999, plus exact bf16 V conversion. Apply the output gate to every
query head as well as the complete tensor; zero vectors need explicit
handling. These are bf16 output gates, not the fp32 GEMV/FFN thresholds.

All72 hardware calls pass across the three builds. At Q24 the actual buffers
plus stack end at51728 bytes/core; `.text` is14400 bytes. This leaves only1984
bytes against the historical16 KiB program budget. Do not infer that a fused
full layer will fit. Hardware runs must be sequential and need device access;
the harness setup is in `../open-wide-deltanet-ab/SKILL.md`.

Exact metrics, resources, artifact hashes and remaining integration blockers:
`specs/open-engine/plans/wide-attention.md`.
