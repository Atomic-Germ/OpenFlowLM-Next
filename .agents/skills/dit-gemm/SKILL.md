---
name: dit-gemm
description: Build, verify and time the open XDNA2 GEMM kernel set for diffusion transformers (dit_gemm, bf16 x bfp16ebs8, 13-18 TFLOPS) and pack weights for it. Use when rebuilding the FLUX.2 klein 4B kernel set, adding a resolution or another DiT family, choosing a datapath for a new large-M GEMM, or debugging dit_gemm output that is NaN, biased, or off by rel_fro ~1.
---

# dit_gemm: the diffusion GEMM

Source is `open_kernels/designs/dit_gemm/`; its README has the full numbers. The exporter
is `open_kernels/export_dit_kernels.py`. Kernel sets are **built, not checked in**
(`src/xclbins/*/open_kernels*/` is gitignored).

## What was learned getting here (don't re-derive it)

1. **Only 8-bit operands are fast on XDNA2.** The MAC array takes int8 or bfp16ebs8 (an
   8-bit mantissa sharing one exponent per 8 values).
   - Plain bf16 in `mm.cc` runs on the fp32 vector unit: ~2.8 TFLOPS, and wider tiles
     don't help.
   - FP8 (E4M3/E5M2) and MX formats have no matmul on aie2p; `aie_api` has them for
     aie2ps only.
   - int8 × int4 (`mmul_8_4`) exists; its rate is unmeasured.
2. **The repo's `gemm_pretiled`** (Whisper, BERT) peaks around 2.9 TFLOPS bf16, 6.1
   bfp16-emulated at tile n=48, and 9-12 TOPS int8 (`utilities/dit-gemm-bench/`).
3. **mlir-aie's asymmetric-tile-buffering config1** claims 24-31 TFLOPS, but only with
   **chess**, which is closed-source and not installed. Its microkernel pins registers
   with `chess_storage()`.
   - Under Peano it spills and returns NaN/inf in ~2/3 of C, even after fixing its
     stack (0xE40 frame vs 0xD00) and its `.bss` quarter counter.
   - **Keep its dataflow and swap in the stock `mm_bfp_mixed.cc` kernel per quarter
     tile.** The dataflow already delivers A in that kernel's layout. That combination
     is `dit_gemm`: correct under Peano at 13-18 TFLOPS.
4. **The stock `mm_bfp_mixed.cc` never sets a rounding mode.** Floor biases both
   in-kernel conversions: rel_fro 8.3e-2 with a −7% mean offset at K=3072.
   `mm_dit.cc` uses round-to-nearest-even: 1.06e-2.
5. **Pack B with round-to-nearest on the host.** The device only decodes. Truncation
   (helper.h's `floatToBfp16`) costs 1.52e-2 end to end against 1.25e-2.
6. **Rebuild kernels from a fresh `final.prj`.** aiecc silently reuses cached kernel
   sources: an edited kernel was compiled from its old copy and measured unchanged.
7. **Quality is fine.** CPU emulation of this exact arithmetic over 8 prompts
   (`utilities/dit-ref/`) scored LPIPS 0.0150 vs bf16, against fp32's own 0.0129.
   int8 W8A8 per-token scored 0.051: it drifts, but is not broken.

8. **SwiGLU epilogue + gathered reads (Phase 3).** Pack the MLP-in weight with
   `pack.interleave_swiglu` (64 gate + 64 up per 128-col tile), set the stream's
   `DG_LAYOUT` `{"epi": {"first_cb": ...}}`; the next GEMM reads `{"a_gather": true}` (the
   first 64 of every 128 columns). Don't try to drop the unused half in the drain: a BD
   stride is at most 2^20 words, so there is nowhere far enough to put it.
9. **Runtime layout** (`lda`, `a_col`, `ldc`, `c_off`, `epi.gap`) is instruction-stream
   only: one xclbin still serves every stream. klein's single-block FU buffer and the text
   encoder's zero-padded hidden (K 2560 out of 3072) use it.

## Rebuild / add a resolution

```
. C:\dev\mlir-aie\iron_env.ps1
python open_kernels\export_dit_kernels.py --resolutions 512,1024
```

- A stream is reused if its `build/<stream>/shape.json` matches; `--force` rebuilds.
- A resolution needs (R/16)² % 512 == 0, otherwise pad M.
- Every stream must share one static xclbin; the exporter refuses otherwise.
- A new family goes in `FAMILIES` plus its own stream function. klein's shapes come from
  diffusers' `Flux2Transformer2DModel`:
  - q/k/v fused to N=3·hidden;
  - single-block `to_qkv_mlp_proj` = 3·hidden + 2·mlp;
  - `to_out` K = hidden + mlp;
  - no biases.

## Verify

```
xrt-smi configure --pmode turbo          # 8.7 vs 13.2 TFLOPS otherwise
python open_kernels\designs\dit_gemm\make_test.py --kernels <set> --stream <s> --out <t>
open_kernels\harness\out\run_kernel.exe <t>\run_<s>.cfg
python open_kernels\designs\dit_gemm\compare.py <t> <s>
python open_kernels\harness\bench.py <t>\run_<s>.cfg --driver open_kernels\harness\out\run_kernel.exe --warm 1
```

Expected results:
- PASS with rel_fro 1.25e-2 (K=3072) to 1.9e-2 (K=12288);
- row cosine > 0.9997;
- 13-18 TFLOPS on image-sized streams.

Diagnosing failures:

| symptom | meaning |
|---|---|
| NaN | kernel codegen or stack |
| rel_fro ~1 | layout: B not from `pack.pack_b`, or a tap/group index |
| a few % with a mean offset | rounding mode |

## Weights

`pack.pack_b(W.T)` takes a torch/diffusers Linear weight transposed to [K, N] and
returns the device buffer, 1.125 bytes per value. klein 4B's block linears come to
~4.2 GB.

Fuse q/k/v (and the single block's q|k|v|mlp-in) by concatenating along N *before*
packing. The model converter mode (`q4nx-build --open-diffusion`) is not built yet.
