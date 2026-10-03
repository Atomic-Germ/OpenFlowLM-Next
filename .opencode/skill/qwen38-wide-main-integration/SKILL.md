---
name: qwen38-wide-main-integration
description: Merge the upstream shared WideDeltaNet and rolled-band changes while retaining the later wide precision modes, then reproduce FFN and full-model regression checks.
---

# Wide primitive upstream integration

Read `specs/open-engine/plans/qwen38-main-oct3.md`. The 2026-10-03 merge
integrates origin/main 0ceb46d, including upstream/main b16e6ab. PR121 contains
our earlier WideDeltaNet work; keep later precision options and reports when
resolving overlap. Combine `band_range` with both compact and ordinary FFN loops.
ROLLED_BANDS contains only qwen35/H2560, not H5120.

The hand-written activation diagnostic is `layer_x/activation_probe.cpp`.
Its old `.cc` suffix overlapped the generated-TU tracking rule. It is not generated.
Worker simulation must load `band_range` as well as `ffn_body`, with ROLLED=False
for ordinary widths. Preserve runtime/catalogue guards and immutable fixtures.

Rebuild with the installed ironvenv (Python3.14, mlir-aie1.4.2, llvm-aie21).
Layer_x builds run sequentially because they share generated translation units.

```bash
wide=open_kernels/designs/wide_deltanet
base="$wide/build_main_oct3"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --ffn-correction --product-correction --block-carry --segment-carry \
  --activation-carry --activation-series --out "$base"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-segmented-dense.py prepare --build-dir "$base/ffn"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/ffn/segmented.cfg"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-segmented-dense.py compare --build-dir "$base/ffn"
```

All26 checks must pass; compare all13 got*.bin arenas with
`build_activation749/compact/ffn`. Reuse the full replay command from
`qwen38-main-integration/SKILL.md`, substituting `build_main_oct3` for
`build_main_merge`. This FFN includes activation-series, so the expected
baseline is `build_activation749/full`: six numerical failures, not two.
Compare every non-symlink binary capture bytewise; do not infer acceptance
from matching tokens. Neural math remains on the NPU.

CPU suite: `ironvenv/bin/python -m pytest specs/open-engine/tests -q`.
Runtime CMake/CTest uses the installed XRT SDK as documented in
`qwen38-upstream-regression/SKILL.md`; this run builds in `/tmp/oflm-main-oct3`.
No distributed host library is replaced. Logs: `/tmp/main-oct3-*.log`.
