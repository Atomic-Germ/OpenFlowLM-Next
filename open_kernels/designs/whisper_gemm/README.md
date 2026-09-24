# whisper_gemm: the open Whisper encoder's GEMMs, seven streams over one xclbin

Spec: `OPEN-WHISPER-KERNELS` in `specs/open-whisper/spec.md`. Issue:
[#72](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/72).

`whisper_gemm.py` is a thin module over the pre-tiled bf16 whole-array GEMM the embedding
models already run on (`npu_offload/gemm_rtp/gemm_pretiled.py`), exactly as
`designs/attn_block/attn_gemm.py` is. Tile (m, k, n) = (64, 64, 32) on 8 columns, runtime
loop bounds, task groups pipelined two deep; A bf16 `[M, K]` row-major, B the `[K, N]`
operand pre-tiled with `npue.tile_b(b, 64, 32, 8, 8, "k,n")`, C fp32 `[M, N]`.

whisper-large-v3-turbo's encoder is seven shapes, with 1500 audio frames padded to 1536:

| stream | M x K x N | GFLOP | per window |
|---|---|---:|---|
| conv1 (im2col, 3 x 128 taps) | 3072 x 384 x 1280 | 3.0 | 1 |
| conv2 (im2col, stride 2) | 1536 x 3840 x 1280 | 15.1 | 1 |
| qkv (Q\|K\|V fused) | 1536 x 1280 x 3840 | 15.1 | 32 |
| o | 1536 x 1280 x 1280 | 5.0 | 32 |
| fc1 | 1536 x 1280 x 5120 | 20.1 | 32 |
| fc2 | 1536 x 5120 x 1280 | 20.1 | 32 |
| xkv (4 decoder layers' cross K\|V) | 1536 x 1280 x 10240 | 40.3 | 1 |

About 2.0 TFLOP per 30 s window.

## Build the set

```
. C:\dev\mlir-aie\iron_env.ps1          # or: source ~/ironenv142/bin/activate
$env:PATH = "C:\Xilinx\XRT;" + $env:PATH
python open_kernels\export_whisper_kernels.py
```

With no `--out` the set goes to `src/xclbins/Whisper-V3-Turbo-NPU2/open_kernels/`, where
the engine finds it with no configuration (the same place the other open engines' sets
live; the build tree's `xclbins` junction and the install step cover it). `--out DIR`
writes elsewhere.

The exporter builds every stream and **refuses the set** unless all seven `final.xclbin`
are the same static configuration. That is not a formality: `fc1` and `xkv` exceed the
drain tiler's stride limit and fall back to one row block per barrier, and `conv2` has
K = 3840. All seven matched conv1 in 70-78 bytes across 11-14 short runs, i.e. UUID and
metadata only.

The same command also builds `<out>/fa/` -- the bidirectional FlashAttention kernel
`src/open_whisper` dispatches to, a SEPARATE hardware context from these seven GEMM
streams, from `../whisper_fa/attn_fa.py` (its own `README.md`). `--no-fa` skips it.

## Test it on hardware

```
python designs\whisper_gemm\make_test.py --set <dir> --out <testdir>
open_kernels\harness\out\run_kernel.exe <testdir>\run_qkv.cfg     # once per stream
python designs\whisper_gemm\compare.py <testdir>
```

Gate: `rel_fro <= 5e-3` and per-row cosine > 0.999, against a float64 reference. Measured
2026-09-19 on Strix (mlir-aie 1.4.2.dev16+g7e00b57, Peano 21.0.0.2026080301): **1.4e-07 to
7.1e-07**, per-row cosine 1.000000000 on all seven.

The test runs from the **exported** set -- one `final.xclbin` plus that stream's
`insts_<name>.bin` -- so it covers what the engine loads, not a private build.

## A note on timing

The harness prints wall clock per dispatch. That is a host-side observation, not an NPU
performance figure. A traced number would need a narrower design: an 8-column design
cannot carry a trace flow (routing runs out), so per-core cycles have to be measured at 2
or 4 columns, which is a different design from the one that ships.
