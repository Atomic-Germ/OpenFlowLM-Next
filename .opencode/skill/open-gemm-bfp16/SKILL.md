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
- Activations: `host::bfp16_pack_a` per GEMM stage (T = 256 tokens, A tile 64 rows). The qkv / o / down outputs are read
  c-tiled in place by the stage after them (`block_host.hpp bfp16_c_row`, 512-byte runs), not untiled into a copy.
- The FFN (OPEN-PREFILL-BFP16-FFN, manifest version 5): gate|up is ONE GEMM, `gemm_bfp16_n16384_k2560_glu`, over the
  interleaved weight `ggu_w` (8-row blocks alternate gate / up; n-tile ct holds ff chunk (ct % 8) * 16 + ct / 8), and its
  epilogue writes silu(gate) * up as bfp16 straight into the H operand that `gemm_bfp16_n3072_k8192_h` reads as A. No host
  stage between them; `Core::glu_phase` runs the two back to back per block.
- Manifest: `gemm_block_variants.bfp16` on the dense layer type (manifest_version 4), selected with
  `OFLM_OPEN_PREFILL_MODE=bfp16` / `open_qwen36_cli --prefill-mode bfp16`. Default stays the bf16 `gemm_block`.
  The route runs layer-major with host attention only (`Core::layer_major_ok`), and refuses to load otherwise.

## Build (Windows)

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1                       # dot-sourced
cd <repo>
python open_kernels\export_qwen36_kernels.py --model-dir <Granite-4.2-3B-NPU2 dir>      # all sets, incl. the four bfp16 ones
python open_kernels\export_qwen36_kernels.py --model-dir <dir> --out <set> --only gemm_bfp16_n4096_k2560,gemm_bfp16_n3072_k2560,gemm_bfp16_n16384_k2560_glu,gemm_bfp16_n3072_k8192_h
```

Each is ~20 s. `gemm_bfp16.py` reads `GB_M GB_K GB_N` (and `GB_TM GB_TK GB_TN`), plus `GB_EPI=glu` (the SwiGLU epilogue, C is
the H operand) and `GB_A=h` (A is an H operand); env names are case-insensitive on Windows, so `GB_M` and `GB_m` are the same
variable -- the tile knobs are spelled `GB_T*` for that reason. `--only` into a copy of a full set rebuilds just these four
and rewrites the manifest (version 5). Engine build: `src/open_qwen36/build.cmd` as in `open-granite-kernels`.

## The glu epilogue: how h gets from a core to down's A without the host

- **The core program is one for every shape and mode**: `rtp[0]` = K chunks between the first and the last (gk/tk - 2),
  `rtp[1]` = epilogue mode, both written by each instruction stream. The last K chunk is peeled so the core still holds its
  A buffer when it calls `gemm_epilogue(C, A, mode)`.
- **Core memory is full** (A 2 x 4608, B 2 x 9216, C 32768, stack 4096 = 64512 of 65536 B), so there is no room for a 4.5 KB
  staging buffer. The held A buffer is exactly one 64 x 64 bfp16 A tile (4608 B): mode 1 encodes h into it through
  `aie::block_vector_output_buffer_stream<bfp16ebs8, 64>` (each push flushes), then copies it into C.
- **C goes out through the memtile's static 8x8-block -> row-major transform** (`dims_to_stream`, the same for every mode).
  Stream word s of a core tile is C's (row s / 128, column s % 128), so the copy puts 32-byte chunk q at C block
  ((q / 16) / 8) * 16 + q % 16, row (q / 16) % 8: the A tile becomes the first 1152 words of the core's 8192-word stream.
  Writing h into the front of C in place is hazard-free, but the transform then scatters it; the A-buffer detour avoids that.
- **The shim cannot split one stream into "A tile here, junk elsewhere"** with one affine pattern, so every core tile is
  written whole, to an H slot: (kc * 4 + row) * 8199 words. 8199 = 9 x 911 keeps each slot offset a whole number of 36-byte
  groups, which a tap in `v8bfp16ebs8` units needs (32 KB slots are not). Down's A tap per row block: `[n-groups (stride 0),
  128 chunks (stride 4 slots), 4, 128 blocks]`. The H buffer is 16.8 MB; 86 % of it is the tiles' unused fp32 tail.
- The column-contiguous n-tile order (ct = (kc % 16) * 8 + kc / 16) is what makes column c's drain one run of slots.
- SiLU is npu-prefill-engine's bf16 hi/lo `swiglu32` (fp32 multiplies are emulated on this core). The result stays an fp32
  accumulator into the bfp16 conversion: rounding it to bf16 first would double-round.

## Verify, in order

1. No NPU: `python -m pytest specs/open-engine/tests/test_prefill_bfp16.py`; `pools_test.exe` (bfp16_gemm and glu hashes),
   `block_host_test.exe` (pack_a hash, untile), `manifest_test.exe` (version 5, the 4-step glu program, 6 refused).
2. One shape on the array: `python open_kernels\designs\gemm_bfp16\make_test.py make --build <build dir> --out <dir> --M 256 --K 2560 --N 4096`,
   `open_kernels\harness\out\run_kernel.exe <dir>\run.cfg`, `make_test.py compare --out <dir>` -> kernel vs decoded-operand
   product rel_fro < 1e-4. The FFN chain: `make_test.py make-glu --build <glu build> --h-build <h build> --out <dir> --K 2560
   --ff 8192 --N 3072` (pass `--N` -- the default 4096 is not down's padded 3072, and the mismatch reads as rel_fro 0.5),
   `run_kernel.exe`, `make_test.py compare-glu` -> H's bytes (>= 99.9 %) and shared exponents (>= 99.99 %) against the host
   encode of silu(g) * u, down(H) rel_fro. A build without `hold()` reads 0.99722 / 0.99715 and FAILs.
3. The model: `utilities\dense-decode-probe\prefill_mode_check.py` (random ids, chat prompts, greedy decode, perplexity, timing).

Every NPU run, timed or not, goes under `dibs run --measure --tag <you> ...` (CLEAN before and after): `dibs` treats one
session's tracks as one holder, so a `dibs run npu` job can land inside another track's timed unit. `lock.py` is gone on
purpose. The harness does not set the power mode; the CLI pins `--pmode performance`.

## Traps

- The mmul probe (`utilities/aie-probes/mmul_rate_probe.*`): `asm volatile("")` crashes Peano's IRTranslator ("unable to
  translate instruction: call"); `aie::mmul<8,8,8,int8,uint4>` does not instantiate.
- A bfp16 block with a saturating +128 mantissa decodes lower than it encoded; re-encoding a decoded operand is not a fixed point.
- **The on-core converter (`to_vector<bfp16ebs8>`) is not the host packer.** When a value of a group rounds to |128| the
  converter raises the shared exponent (for -128 too); `bfp16_encode_block` saturates at 127 (lower error for the other seven).
  Measured on the glu harness: 746 of 262,144 blocks, 0.28 % of H's bytes. `aie::set_saturation` (saturate or symmetric)
  changes nothing. **The fix is per element, no cross-lane work** (`hold()` in `mm_bfp_acc.cc`): only a value in the group's top
  binade can round to 128, and it does exactly when its significand is >= 255/128, so set such a value's mantissa field to
  0x7E0000 (127/64) before converting. In the top binade that encodes to 127, the host's saturation; in a lower binade it rounds
  to the same mantissa as before. Result: 5 of 262,144 exponents differ (SiLU's 2^-15 across a power of two), 99.95 % of bytes
  equal (a negative extreme is -127 where the host has -128), no measurable cost. Dead ends on the way: a per-group max exponent
  from `shuffle_down` maxima wrote wrong exponents in 80 % of blocks and tripled the dispatch; converting h's powers of two to
  read the converter's exponent crashed Peano's clang (a `block_vector_output_buffer_stream` over a stack array).
- The engine's gate|up dispatch may be slower than the harness's: in controlled units it blocks 1.72 ms at pp512 and 2.09 ms at
  pp2048 (plus ~0.15 ms of pack under it); untimed `run_kernel` runs read 1.56-1.71 ms alone. Not established, cause unknown. An
  untimed thread-count probe that read 2.14 ms at 24 OpenMP threads and 1.63 at 6 ran while other tracks' NPU jobs could overlap
  it (`dibs` grants one session's tracks the NPU together), so it is not evidence for a power or DDR explanation.
- Hiding host work under the dispatches did not pay: an o + FFN pipeline (o a block ahead, o's residual, norm and gate|up pack
  under the next dispatch) was bit-identical and no faster in two interleaved units (754 against 754 at pp512); it is not kept.
- **Measure with a control.** Every timed unit carries a cell whose value is known (this route's pp512, median ~850 tok/s on
  2026-10-15), and a unit whose control is more than 5 % off is rerun; `ab_bench`-style interleaving alone did not catch an
  overlapped unit that read the route at 768. Run untimed NPU work under `dibs run --measure` too, never beside a timed unit.
- **One context, one core program: the context's xclbin is the first step's build** (`contexts.gemmbfp` = the qkv stream's).
  After any change to the core program, rebuild every bfp16 stream, or the new instruction streams run on the old program.
- `Core::wait_run` clears `Inflight::k`; read the kernel's name first (a segfault only with `OFLM_OPEN_DISPATCH_LOG=1`).

## Result (2026-10-15, CLEAN with a control, the glu FFN + hold)

pp512 / 1024 / 2048, tok/s min / median: five-step 501 / 537, 423 / 423, 231 / 268 -> glu FFN 819 / 858, 549 / 554, 317 / 332
(1.60x / 1.31x / 1.24x); the default bf16 route in the same kind of unit 239 at pp512 and 187 at pp2048 (medians). At pp512
what is left: host attention 199 ms, the GEMM waits 212 (gate|up 137), residual + norm 86, packs 34, qkv's copy 29, syncs 14.
Accuracy against the bf16 route, with `hold()`: perplexity ratio 0.9964, KL 0.00267 (the five-step route's 0.00276), random-id
corr min 0.9963; two per-item extremes of OPEN-PREFILL-BFP16 still missed (one near-tie greedy continuation of six, one flip at
a 0.209 margin), on other prompts than without `hold()`. Memory: +1.7 GiB peak over the default route. Numbers and procedure:
OPEN-PREFILL-BFP16-FFN.

## Result (2026-10-13/14, no gate, FrameServer idle context present)

GEMM at 256 tokens, 10.6-12.1 TFLOPS on the four Granite shapes, rel_fro 0 against the decoded-operand product. Logits against the
bf16 route: corr min 0.995 (mean 0.9996), 80-token greedy identical on six prompts, perplexity ratio 0.9975. pp512 229 -> 434
tok/s, pp2048 144 -> 227. The route is now host-bound (GEMMs 167 of 1307 ms at pp512). Numbers and procedure: OPEN-PREFILL-BFP16.
The mmul probe says int8 8x8x8 is 0.56 ns an mmul against 4.4 (bf16 4x8x8) and 3.5 (bfp16 in an unpipelined loop): an int8
GEMM remains the faster ceiling, if the per-32-k rescale or a per-row weight requantisation can be paid for.
