---
name: open-wide-model-precision
description: Build compensated K6144 output, FFN and RMSNorm probes, precise wide attention, and replay the real 64-layer Qwen38 model against unchanged reference fixtures.
---

# Full-depth wide model precision

Read `specs/open-engine/plans/qwen38-model-precision.md` for measured acceptance,
including unsuccessful intermediate variants. Primitive acceptance alone does
not establish full-model or runtime support.

Use the existing `Models/qwen38-27b/converted` container and immutable reference
fixtures in `open_kernels/designs/wide_deltanet/build_full_model`. Download and
conversion instructions are in `open-qwen38-model`; do not regenerate intact
weights or references to investigate a numerical failure.

## Build the isolated kernels

Use `ironvenv`. Build sequentially: the layer_x generator shares its generated
translation units. Keep the baseline kernel directories intact.

```bash
base=open_kernels/designs/wide_deltanet/build_model_precision
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 6144 --projection-n 5120 --projection-correction --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --ffn-correction --out "$base"
ATTN_PROBE_PRECISE=1 PATH=/opt/xilinx/xrt/bin:$PATH ironvenv/bin/python \
  open_kernels/build_design.py open_kernels/designs/attn/wide_probe.py "$base/attention"
LN_N=5120 LN_STREAM_COMPENSATED=1 PATH=/opt/xilinx/xrt/bin:$PATH ironvenv/bin/python \
  open_kernels/build_design.py open_kernels/designs/ln/ln.py "$base/ln"
```

The output-only probe releases unused recurrent scratch (ds1280 ->32 floats).
Its corrected K6144 table is27008 bytes, total placement62976 bytes/core and
text3792 bytes. This does not establish that a fused whole-layer core fits.

The corrected FFN retains actual main-core scratch and uses K4096 down segments:
4096+4096+4096+4096+1024. The existing pool bytes, DMA slice law and accumulation
worker remain in use. The K5120 up/gate table sets the22592-byte maximum;
placement63552 bytes/core, text13840 bytes. The default uncorrected recipe still
uses8192+8192+1024. Both modes retain the same BF16 activation boundaries.

Attention's opt-in precise mode improves Q/K normalization, RoPE and final
normalization/gating. The existing online softmax and query storage remain in
use. The probe restricts this mode to Q24/KV4/HD256/ROT64 and hashes the precise
math header and mode. Defaults retain the existing arithmetic.

Compensated streamed LN stores64 rather than32 statistics floats, resetting
compensation at the first half. It occupies57600 bytes/core including stack;
text3872 bytes. The mode is opt-in and rejects non-streamed widths.

## Primitive gates and unchanged full-model replay

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-dense-projection.py prepare --build-dir "$base/projection_k6144"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-segmented-dense.py prepare --build-dir "$base/ffn"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-attention.py prepare --build-dir "$base/attention"
ironvenv/bin/python utilities/test-wide-ln.py prepare --build-dir "$base/ln"
```

Run `projection.cfg`, `segmented.cfg`, `attention.cfg` and `ln.cfg` sequentially through
`open_kernels/harness/out/run_kernel`, then run each utility's `compare` stage.
Set `HARNESS_TIMEOUT_MS=30000` and the XRT runtime library path as needed.
The harness build is documented in `open-wide-decode`; prefer a persistent
ignored build directory because `/tmp` is cleared on reboot.

Only after all selected primitive comparisons pass:

```bash
ironvenv/bin/python utilities/replay-wide-model.py \
  --source open_kernels/designs/wide_deltanet/build_full_model \
  --out "$base/full_ln" --output-projection "$base/projection_k6144" \
  --ffn "$base/ffn" --attention "$base/attention" --ln "$base/ln"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/full_ln/decode.cfg"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-full-model.py compare \
  --out "$base/full_ln"
```

Replay requires a fresh destination, verifies source and replacement artifact
hashes, links only immutable fixtures, and changes only kernel bindings. Device
captures are separate files. Keep the source directory available. Omit `--ffn`
or `--attention`/`--ln` for an ablation. The existing comparator remains authoritative;
do not substitute conditional references, feedback IDs or relaxed tolerances.

Run `ironvenv/bin/python -m pytest specs/open-engine/tests -q` after source edits.
Record numerical failures as failures even when tokens and reset replay match.

Recorded status: none of the five full-model variants passes all inherited
checks. Corrected output alone leaves two failures (layer63 residual and final
norm on token1). All modes together leave ten. Do not default to all switches.
Use `utilities/diagnose-wide-model-rounding.py --out <replay>` to distinguish
local norm error from propagated inputs. The first changed BF16 value in
`full_ln` is layer1/channel2931; its norm is locally exact, while the preceding
residual differs by4.6566e-9. Investigate corrected Q4 block products/reductions
before making another norm-only change.
