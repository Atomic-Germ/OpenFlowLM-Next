---
name: open-gemm-bfp16
description: Build, test and measure the opt-in bfp16 block prefill route (gemm_bfp16, OFLM_OPEN_PREFILL_MODE=bfp16) for Granite on the dense recipe. Use when rebuilding its four GEMM xclbins, adding another dense family to BFP16_FAMILIES, debugging a bfp16 route that returns NaN, garbage or a wrong shape, or comparing it with the bf16 route.
---

# The bfp16 prefill GEMM

`open_kernels/designs/gemm_bfp16/` is a whole-array bfp16 x bfp16 -> fp32 GEMM, C[M, N] = A[M, K] B[N, K]^T, derived
from ggml-xdna's `kernels/bfp16_gemm` (Apache-2.0 WITH LLVM-exception; its MIT host packers are `src/open_qwen36/bfp16.hpp`
and `designs/gemm_bfp16/pack.py`). Buffer arguments are weight, x, y, the order every GEMM step uses.

## Why bfp16 and not int8

On aie2p only 8-bit operands reach the fast MAC array (see the `dit-gemm` skill). `aie_api` has int8 x int8 8x8x8
(`mmul_8_8.hpp`) and bfp16ebs8 x bfp16ebs8 8x8x8 (`mmul_bfp16_bfp16.hpp`), both 512 MACs an instruction, against the
4x8x8 = 256 of the native bf16 `mmul_bf16_bf16.hpp` that `gemm_q4_prefill` runs. The Q4_1 weight has a scale and a min per
(row, 32 k): an int8 product needs either a rescale of the int32 sums every 32 k, which costs as much as the mmul, or the
weights re-expressed as int8 under one scale per row, a second quantisation. bfp16ebs8 carries a shared exponent per 8 k
for both operands, so the exponent math is in the hardware and the activations keep their outliers (measured round trip,
Gaussian: 0.6 % per operand). ggml-xdna's own log (`kernels/bfp16_gemm/results.log`, quiet box) has this tile at 8.8-12 TFLOPS for M = 256 tokens, K 2048-6144, N 2048-12288, against 2.2-2.4 for `gemm_q4_prefill`.

## What the route is

- Weights: the container's q4_1 chunks, dequantised to float and encoded to bfp16 at load by the `bfp16_gemm` pack op
  (`pools.cpp bfp16_gemm_pack`, twin `recipes/pack.py pack_bfp16_gemm`). Rows are padded to a multiple of 1024
  (`BFP16_ROWS`: tile n 128 x 8 columns) with zero blocks; a stacked weight (q|k|v) writes each tensor at `row0` of the
  `n_total`-row weight. No q4nx-build change: the converted model is the same file, the repack is the engine's.
- Activations: `host::bfp16_pack_a` per GEMM stage (T = 256 tokens, A tile 64 rows), output untiled to [N, T]
  (`bfp16_untile_c`) so the stages after the GEMM are unchanged.
- Manifest: `gemm_block_variants.bfp16` on the dense layer type (manifest_version 4), selected with
  `OFLM_OPEN_PREFILL_MODE=bfp16` / `open_qwen36_cli --prefill-mode bfp16`. Default stays the bf16 `gemm_block`.
  The route runs layer-major with host attention only (`Core::layer_major_ok`), and refuses to load otherwise.

## Build (Windows)

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1                       # dot-sourced
cd <repo>
python open_kernels\export_qwen36_kernels.py --model-dir <Granite-4.2-3B-NPU2 dir>      # all sets, incl. the four bfp16 ones
python open_kernels\export_qwen36_kernels.py --model-dir <dir> --only gemm_bfp16_n4096_k2560,gemm_bfp16_n3072_k2560,gemm_bfp16_n8192_k2560,gemm_bfp16_n3072_k8192
```

Each is ~20 s. `gemm_bfp16.py` reads `GB_M GB_K GB_N` (and `GB_TM GB_TK GB_TN`); env names are case-insensitive on Windows, so
`GB_M` and `GB_m` are the same variable -- the tile knobs are spelled `GB_T*` for that reason. Engine build:
`src/open_qwen36/build.cmd` as in `open-granite-kernels`.

## Verify, in order

1. No NPU: `python -m pytest specs/open-engine/tests/test_prefill_bfp16.py`; `pools_test.exe` (bfp16_gemm hash),
   `block_host_test.exe` (pack_a hash, untile), `manifest_test.exe` (version 4).
2. One shape on the array: `python open_kernels\designs\gemm_bfp16\make_test.py make --build <build dir> --out <dir> --M 256 --K 2560 --N 4096`,
   `open_kernels\harness\out\run_kernel.exe <dir>\run.cfg`, `make_test.py compare --out <dir>` -> kernel vs decoded-operand
   product rel_fro < 1e-4.
3. The model: `utilities\dense-decode-probe\prefill_mode_check.py` (random ids, chat prompts, greedy decode, perplexity, timing).

Every NPU run goes through `python ~/.claude/locks/lock.py timing --label <you> -- <cmd>`. The gate waits for a quiet
machine and treats *any* other NPU context as a rival: **an open camera app keeps the Windows `FrameServer` svchost holding
an idle realtime context, and the gate then never opens.** Close the camera app, or tell the user.

## Traps

- The mmul probe (`utilities/aie-probes/mmul_rate_probe.*`): `asm volatile("")` crashes Peano's IRTranslator ("unable to
  translate instruction: call"); `aie::mmul<8,8,8,int8,uint4>` does not instantiate.
- A bfp16 block with a saturating +128 mantissa decodes lower than it encoded; re-encoding a decoded operand is not a fixed point.
