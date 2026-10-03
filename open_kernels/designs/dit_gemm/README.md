# dit_gemm: the diffusion transformer's GEMM, bf16 × bfp16ebs8 → bf16

`dit_gemm.py` computes `C[M,N] = A[M,K] @ B[K,N]`:
- A: bf16 activations, row-major.
- B: weights, bfp16ebs8, pre-packed by `pack.py`.
- C: bf16, row-major.

It carries every linear layer inside FLUX.2 [klein] 4B's transformer blocks. It also fits
any large-M GEMM whose shape meets its constraints, so dense LLM prefill is a candidate.

- **Dataflow:** mlir-aie's asymmetric-tile-buffering GEMM (Wang et al., arXiv:2511.16041).
  Each of the 32 cores keeps a 128×128 bf16 C tile resident across the whole K walk and
  takes A 32 rows at a time.
- **Microkernel:** mlir-aie's stock bf16 × bfp16 kernel (`mm_dit.cc`). Its header says why
  not the example's own kernel.
- **Toolchain:** everything builds with Peano. Nothing needs chess.

Why bfp16: on XDNA2 (aie2p), only 8-bit operands reach the fast MAC array.
- int8 and bfp16ebs8 (an 8-bit mantissa sharing one exponent per 8 values) are the fast
  paths.
- bf16 matmul either converts to bfp16 in-core or falls back to the fp32 vector unit
  (~2.8 TFLOPS measured).
- FP8 (E4M3/E5M2) and MX formats have no matrix unit on this generation; `aie_api` has
  those only for aie2ps.

## Numbers (2026-09-26, Ryzen AI 9 HX 370, turbo, `open_kernels/harness/bench.py`, median)

klein 4B at 1024² (4096 image + 512 text tokens), one xclbin, one hardware context:

| stream | M×K×N | ms | TFLOPS | rel_fro vs exact |
|---|---|---:|---:|---:|
| r1024_sgl_in | 4608×3072×27648 | 59.42 | 13.2 | 1.25e-2 |
| r1024_sgl_out | 4608×12288×3072 | 20.25 | 17.2 | 1.88e-2 |
| r1024_img_qkv | 4096×3072×9216 | 13.05 | 17.8 | 1.25e-2 |
| r1024_img_out | 4096×3072×3072 | 4.70 | 16.5 | 1.25e-2 |
| r1024_img_ffin | 4096×3072×18432 | 33.92 | 13.7 | 1.25e-2 |
| r1024_img_ffout | 4096×9216×3072 | 20.84 | 11.1 | 1.70e-2 |
| txt_qkv / txt_out / txt_ffin / txt_ffout | 512×… | 3.87 / 3.39 / 3.67 / 2.28 | 7.5 / 2.9 / 15.8 / 12.7 | 1.25e-2 - 1.69e-2 |

Per denoising step: 5 double blocks × 85.7 ms + 20 single blocks × 79.7 ms ≈ **2.0 s of
GEMM**, so **~8 s per 4-step 1024² image**. Attention, norms, modulation and the VAE are
extra.

Error is bfp16 rounding of both operands plus the accumulator's re-rounding to bf16 every
64 of K, which is why it grows with K. At image level this arithmetic is indistinguishable
from bf16: LPIPS 0.0150 against fp32's 0.0129 noise floor (`utilities/dit-ref/`).

## Build, export, test

```
. C:\dev\mlir-aie\iron_env.ps1
python open_kernels\export_dit_kernels.py                  # klein 4B, 512 + 1024 -> src/xclbins/FLUX.2-klein-4B-NPU2/open_kernels/
python open_kernels\designs\dit_gemm\make_test.py --kernels <set> --stream r1024_sgl_in --out <testdir>
open_kernels\harness\out\run_kernel.exe <testdir>\run_r1024_sgl_in.cfg
python open_kernels\designs\dit_gemm\compare.py <testdir> r1024_sgl_in
python open_kernels\harness\bench.py <testdir>\run_r1024_sgl_in.cfg --driver open_kernels\harness\out\run_kernel.exe --warm 1
```

The exporter refuses the set unless every stream's `final.xclbin` is the same static
configuration. Loop bounds are runtime parameters, so only the instruction streams
differ.

Compare gate:
- finite everywhere;
- rel_fro ≤ 3e-2 against bf16 A × bf16 B;
- row cosine > 0.999.

Layout or dataflow bugs land at rel_fro ~1 or produce NaN, far outside that.

## SwiGLU epilogue and runtime layout (Phase 3, 2026-09-26)

A stream's `DG_LAYOUT` (JSON, runtime sequence only — every stream still shares one xclbin):

| key | effect |
|---|---|
| `epi: {first_cb, gap}` | column groups (1024 columns) from `first_cb` get the SwiGLU epilogue, and land `gap` columns further right in C |
| `a_gather` | A's K columns are the first 64 of every 128 of the source (the epilogue's output) |
| `lda`, `a_col`, `ldc`, `c_off` | row strides / offsets, e.g. K = 2560 read out of a 3072-wide zero-padded buffer |

- **Epilogue.** The weight's MLP columns are packed 64 gate + 64 up per 128-column tile
  (`pack.interleave_swiglu`). After the K walk the core turns the tile's first 64 columns
  into `silu(gate) * up` in place (`dit_swiglu_epi`, 416 B of code). The tile leaves
  whole: the memtile's output pattern is static, so it can't be compacted.
- **Gathered A.** The next GEMM reads only the valid halves: one 4-D descriptor per
  strip, `[K/512, 128, 8, 64]` with strides `[1024, lda, 128, 1]`. This replaces the
  separate SwiGLU pass (read 2×, write 1× the MLP activations) with nothing.
- **klein's single blocks** leave a `gap` of 6144 columns before the MLP tiles. dit_fa
  writes the attention output into it in the same 64-of-128 pattern, so the
  out-projection gathers `[attention | SwiGLU]` with one uniform pattern. No concat
  buffer, no SwiGLU pass.
- **Numbers** (512 rows): epilogue 512×3072×18432 in 3.2 ms, PASS at rel_fro 2.2e-2
  (SiLU amplifies the GEMM error a little); gathered 512×9216×3072 in 2.5 ms, PASS at
  1.7e-2. In the chain tests (`utilities/dit-chain/`) the epilogue reproduces dit_ew's
  SwiGLU bit for bit.
- **Why not route the unused half to scratch in the drain:** a BD stride is at most 2^20
  32-bit words (4 MB), so the scratch can't sit a buffer's length away. The same limit
  forced dit_fa to fill wide K/V buffers in 32-row blocks.
- The explicit A fill (`[K/512, 128, 512]`, replacing TensorTiler taps) also made small
  shapes faster: 512×3072×3072 went 3.4 → 1.0 ms. Re-time the full set before quoting it.

## Constraints and traps

- **Shapes.** M % 512, K % 512, N % 1024. A resolution whose image-token count is not a
  multiple of 512 (e.g. 768² = 2304) needs M padded.
- **Set turbo mode first** (`xrt-smi configure --pmode turbo`). This design measured
  8.7 vs 13.2 TFLOPS in performance vs turbo.
- **Stale kernels.** aiecc reuses kernel sources cached in `final.prj`.
  `build_design.py` deletes it; anything else that builds this design must too.
- **The stack is 0xF00.** Peano's frames here are larger than chess's; the example's
  0xD00 overflowed into C with its own kernel.
- **B must come from `pack.pack_b`.** It is what the tests exercise, and a
  hand-rolled layout fails at rel_fro ~1.
