# dit-gemm-bench: whole-array GEMM throughput at diffusion-transformer shapes

Phase 0 of the image-diffusion work. It answers one question before any diffusion code
exists: how fast can this NPU run the large-M GEMMs a DiT denoise step is made of? At
1024x1024, M = 4096 image tokens and K/N = 3072-12288.

**Best route so far: `atbs`**, 13-17 TFLOPS on every DiT shape (median 13.1-15.4), built
entirely with Peano. It is mlir-aie's asymmetric-tile-buffering dataflow (a 128x128 C tile
per core, a quarter of A per call) driving mlir-aie's stock bf16 x bfp16 microkernel
(`atb/mm_atb_stock.cc`) in place of config1's chess-tuned one. At 15.6 ms for
4096x3072x9216 it moves ~556 MB, ~36 GB/s -- near the ~45 GB/s shim roof, so the next gains
are in bytes (bfp16 activations, fusion), not the kernel.

| script | measures |
|---|---|
| `dit_gemm_bench.py` | `npu_offload/gemm_rtp/gemm_pretiled.py`, the GEMM Whisper, block attention and the BERT embedders ship, on every datapath it has: bf16, bf16 with bf16 C, bfp16-emulated, int8, plus wider-tile variants |
| `bfp_gemm_bench.py` | mlir-aie's bf16 x bfp16 designs, copied here: `wam/` (symmetric whole-array, 64x64x64), `atb/` (asymmetric tile buffering, 128x64x128, upstream kernel) and `atbs` (the same dataflow with the stock kernel) |

Both scripts need a shell where `C:\dev\mlir-aie\iron_env.ps1` has been dot-sourced.
`bfp_gemm_bench.py` also needs `open_kernels\harness\out\run_kernel.exe` (see
`open_kernels/harness/README.md`; `build.cmd` with `XRT_INCLUDE_DIR` and `XRT_LIB_DIR`
set).

```
python utilities\dit-gemm-bench\dit_gemm_bench.py --out results.json
python utilities\dit-gemm-bench\bfp_gemm_bench.py --design wam --shape 4096x3072x9216 --rounding nearest
```

`dit_gemm_bench.py` sets `xrt-smi configure --pmode turbo` for the run and restores
`performance` afterwards. For `bfp_gemm_bench.py`, set turbo mode yourself first. It
matters: `atbs` at 4096x4096x2048 measured 8.74 TFLOPS in performance mode and 13.24 in
turbo.

**Timing.** All times are start->wait wall clock per dispatch, the minimum over warm
runs. That is the same observation `open_kernels/harness` prints. It is a dispatch
figure, not a traced cycle count.

## Results (2026-09-25, Ryzen AI 9 HX PRO 370, NPU driver 32.0.20102.3930, turbo)

TFLOPS (TOPS for int8) at M = 4096:

| datapath | o 3072x3072 | qkv 3072x9216 | mlp_down 9216x3072 | q21 4096x12288 | rel. error |
|---|---:|---:|---:|---:|---|
| pretiled bf16, fp32 C (Whisper today) | 2.54 | 2.83 | 2.84 | 2.85 | 5e-7 |
| pretiled bf16, bf16 C | 2.82 | 1.99 | 1.99 | 1.99 | 1.7e-3 |
| pretiled bf16, tile n=48 | 2.86 | 2.87 | | | 1.7e-3 |
| pretiled bfp16-emulated, n=32 | 4.07 | 4.24 | 4.04 | 4.90 | 9.6e-3 |
| **pretiled bfp16-emulated, n=48** | **6.13** | **5.93** | | | 9.6e-3 |
| **pretiled int8 (64,64,64)** | **9.33** | **9.78** | **11.86** | **11.04** | 1.7e-3 * |
| mlir-aie `wam` bf16 x bfp16 (Peano) | 5.30 | 4.61 | does not lower | 3.93 | 1.25e-2 |
| **`atbs`: ATB dataflow + stock kernel (Peano)** | **17.08** | **14.89** | **13.70** | **15.19** | 1.25e-2 - 1.70e-2 |
| mlir-aie `atb` (Peano) | | | | | output is garbage |
| mlir-aie `atb` (chess), upstream's figure | | | | | 24.3 at 4096x4096x2048 |

\* The int8 inputs here are already integers, so 1.7e-3 is only the bf16 rounding of C. The
error from quantizing real activations to int8 is not measured.

Control: `attn_block`'s 2048x256x2048 takes **2.04 ms** today, both through this script
and through `run_kernel.exe` (rel_fro 1.1e-7). Its README records 0.95 ms. The recorded
figure does not reproduce on this box.

## What the numbers say

- **Plain bf16 is capped near 2.9 TFLOPS by the kernel.** Without bfp16 emulation,
  `mm.cc` does the bf16 matmul on the fp32 vector unit, not the MMAC. Wider tiles buy
  nothing (n=48: 2.86 vs 2.54), and DDR runs at only 20-24 GB/s.
- **Every datapath that uses the MMAC benefits from a wider tile.** bfp16-emulated goes
  from n=32 to n=48 and gains 1.45x. The cores' two 32-bit input streams cap a core at
  4*m*n/(m+n) MAC/cycle on 2-byte operands, which predicts 1.28x.
- **bf16 C hurts once N > 4096.** Above that the drain falls back to one row block per
  group (the DMA-stride guard), and the narrowing no longer pays for itself.
- **int8 is the fastest path the repo already has**, at 3.3-4.2x plain bf16. Its cost is
  activation quantization, which diffusion has not been tested against.
- **mlir-aie's asymmetric-tile-buffering design is the ceiling**: 24-31 TFLOPS upstream
  on this exact chip. It needs **chess, which is closed-source** and not installed here.
  See below for its status under Peano.

## Traps found on the way

1. **Stale kernels.** aiecc keeps kernel sources and objects in `final.prj` and silently
   reuses them. An edited kernel was compiled from its old copy (it kept the `Sep 18`
   mtime). `bfp_gemm_bench.py` deletes `final.prj` before every build, as
   `open_kernels/build_design.py` does.
2. **Rounding mode.** mlir-aie's stock `aie2p/mm_bfp_mixed.cc` never sets a rounding
   mode, so floor is in force. Both of its conversions (A bf16 -> bfp16, and the
   accumulator -> bf16 after every K step) are biased toward -inf. At K = 3072 that
   costs **rel_fro 8.3e-2 with a -7% mean offset**. `wam/mm_bfp_mixed.cc` switches to
   round-to-nearest-even: **1.06e-2**, same speed. `mm.cc`'s emulation path already does
   this.
3. **Host rounding of B.** helper.h's `floatToBfp16` truncates. `--rounding nearest`
   encodes weights round-to-nearest, which the device decodes identically: 1.52e-2 ->
   1.25e-2 end to end.
4. **Wide N does not lower.** Upstream `whole_array_mixed` fails to lower above N = 4096
   ("Stride 3 exceeds the [1:1048576] range" on the C drain). `wam/` carries
   gemm_pretiled's one-row-block guard.
   - K = 9216 still fails, on B's fill: a 9-byte bfp16 block puts the stride over the
     limit in 32-bit words. The fix is a host reorder that makes each column's n-tiles
     contiguous. Not done.
5. **ATB under Peano -- solved by swapping the kernel, not fixing it.** The stack and
   `.bss` counter below were real, but after fixing both, config1's kernel still returns
   NaN/inf in ~2/3 of C. The pattern repeats every quarter-tile call and grows toward
   higher columns: codegen, most likely Peano spilling the twelve 9-byte bfp16 vectors
   chess pins in registers. The dataflow already hands A to the core in the stock
   kernel's layout, and C in the same blocked layout, so `atbs` runs the stock kernel on
   each quarter tile, re-packs B on the host (`atbs_layout`), and is correct.
   Original notes on the upstream kernel:
   **ATB's stack under Peano.** Upstream's kernel pins registers with
   `chess_storage(...)`, which Peano ignores. Peano spills and gives the matmul a
   **0xE40-byte frame against a 0xD00 stack**. `atb/` raises the stack to 0xF00 (L1
   still fits). The output is **still NaN/1e36 garbage** with upstream's own integer
   test data, so something else in the Peano build is wrong. Not yet diagnosed. Its
   10-14 TFLOPS timings mean nothing until C is right.
6. **Wider isn't better in `wam`.** Wider or shallower tiles are slower there: 64x32x128
   ran at 1.56 and 64x64x96 at 3.63, against 5.30 at 64x64x64. The kernel's loop
   structure decides that design, not stream bandwidth.

## Closed-source requirements

- **chess (xchesscc)**: mlir-aie's `gemm_asymmetric_tile_buffering` configs declare
  `use_chess=True`, and upstream's 24-31 TFLOPS figures are chess builds. It is not
  installed here and is not open. `atb/n32_core_atb.py` defaults to Peano and takes
  `use_chess` as a switch.
