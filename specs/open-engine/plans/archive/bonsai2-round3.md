# Plan: Bonsai 2 round 3 (integer dequant in the prompt GEMM, integer FWHT in decode)

**Status:** done (2026-10-06). Builds on `bonsai2-ternary.md` (PR #161).

## Spec impact

**Modified requirements**
- **OPEN-GEMM-T2:** a t2 spec's GEMM builds add `GQP_IDQ=1` (build directories `_t2kyxi`): the
  dequant builds the weight's bfp16 blocks with integer operations, as planes the matmul loads
  directly. y is unchanged to the bit. Verification: `test` for the build env and directories
  (`tests/test_ternary_hadamard.py`); `manual` for the kernel (`xbfp_probe.py` variants `xbfp` /
  `idq`) and the whole model.
- **OPEN-HADAMARD:** the t2 decode preps, the down projection's prep core and the rotated lm head's
  prep take the transform in integers straight into the per-128 table (`wht_int.h`). Not bit-exact
  against the fp32 transform; bounded at 2^-10.8 relative RMS against fp64. Verification: `manual`
  (the `designs/fwht_int` probe against `fwht_ref.py`'s model, the token gates, the 40-token check).
- **OPEN-QUANT-T2:** the rotated lm head's prep and its standalone test's reference follow the
  integer transform.
- **OPEN-DECODE-ONE-CONTEXT-DENSE:** the prep core builds the down projection's second piece's
  whole table; the main cores copy it in (`dense_prep_f32` with K == 0).

## Measurement history

**The integer transform, first build (2026-10-05, `open_kernels_fwht`).**
1. Every probe table bit-exact against the model, both paths. Error against fp64 `H x / 32`,
   relative RMS over the probe's blocks: integer 5.3e-5 (bf16) and 6.7e-5 (fp32); fp32 path 1.7e-3.
   On synthetic residual-like inputs with 1000x outliers the integer path's error is 2.8e-4 to 4e-4.
   Time per block at the decode's `-Oz` (CLEAN window): integer 1.41 us (bf16) and 1.51 us (fp32),
   against 2.80 and 2.84 us for the fp32 path. A first version kept the intermediate in int32 and
   multiplied int16 x int32 (2.54 us): that multiply runs far below the int16 rate. `-Os` gives the
   passes hardware loops (1.41 -> 1.27 us a block) but the step time came out the same for 96 B more
   program, so the preps stay at `-Oz`.
2. 33/33 and 129/129. Logits against `open_kernels_r2` over 164 files: Pearson r min 0.99796, mean
   0.99993; top-1 the same in all 164. Main cores at 15936 of 16384 B (13760 before). Same-window A/B
   against `open_kernels_r2` (prompt512.ids `--gemm-block` + 128 steps, 3 rounds, CLEAN): decode
   159 / 158 / 159 -> 157 / 157 / 157 ms/token, step minimum 156.3 / 155.8 / 155.7 -> 153.9 / 153.8 /
   154.2 ms.
3. The first 40 greedy tokens on short.ids decode to text that PrismML's llama.cpp greedy output
   (Vulkan, 2026-10-02) begins with.

**The shipped build (2026-10-06, `open_kernels_fwht2`: one quantizer for bf16 and fp32 inputs, and
piece 1's transform and table on the prep core in integers).** Logits against `open_kernels_fwht`
r min 0.99857 (piece 1's numerics changed). Decode 164 / 159 / 160 -> 158 / 157 / 159 ms/token
against `open_kernels_r2`, step minimum 159.7 / 155.8 / 155.9 -> 154.3 / 154.4 / 154.5 ms. Main
core `.text` 15936 (fwht) -> 15440 -> 15024 B.

**The integer dequant (2026-10-05).** Kernel probe, timing lock, 4 interleaved rounds, min: 1.799 ->
1.478 ms a dispatch; streams alone 1.243, the new build without its matmul 1.262 and without its
dequant 1.325 (the production build: 1.298 and 1.316). The old dequant was bounded at ~8 cycles a
block by the store stream's push / flush chain. Whole model, interleaved A/B under the timing lock,
3 rounds: the GEMM stage 3.33-3.42 -> 2.85-2.99 s; decode unchanged at 159-160 ms/token.

**Program memory, merged main cores.** 15360 B at the earlier 2560 B chunks; 15248, then 15280 with
the tail, the row-by-row pass 2 and the prep rewrite; 13760 with the data-word counts and the prep
core's K < 0 path; 15936 with the integer transform alone; 15024 with piece 1's table from the prep
core.
