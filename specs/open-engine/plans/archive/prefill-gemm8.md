# Dense prefill on 8-bit GEMMs (dit_gemm)

**Status (2026-10-10): done** (OPEN-PREFILL-GEMM8: gate PASS, 981 tokens 11.1 -> 4.7 s, +3.5 GiB). Design notes and Phase 0 measurements:
`C:\code\openflowlm-dense\.claude\plans\prefill-gemm-8bit.md`.

## What changes

Qwen3-8B's dense block route spends 7.4 s of a 981-token prefill (11.6 s) in its five
projection GEMMs, which multiply bf16 by bf16 at ~1.9 TFLOPS. `dit_gemm` (from #136) feeds
the matrix unit 8-bit operands (bf16 activations x bfp16ebs8 weights) and measured 17-19 TFLOPS
at these shapes. It needs its own weight copy (9 bits a weight) and blocks of 1024 tokens.

- **The route.** A dense set whose projection shapes fit `dit_gemm` (K % 512, N % 1024) and
  whose attention runs as NPU products (`attn_block.prep`) gets a second route at t = 1024:
  the same five steps on `dit_gemm` streams in one `dit` context.
- **The weights.** A new pack op, `bfp16_dit`, converts a projection's q4_1 chunks to
  `dit_gemm`'s B layout at load (`pack.pack_b`, the same bytes in C++ and NumPy). The copy is
  made only when the route is chosen. On Qwen3-8B it is ~7.8 GB on top of the q4_1 pools the
  decode path still needs.
- **The switch** is the existing `--prefill-mode` (OPEN-PREFILL-MODE). The 8-bit route is
  `gemm_block` (`fast`, the default) and today's q4_1 route is the `lean` variant.
  Default-on is conditional on the accuracy gate below; if it fails, the recipe swaps the two.
- **Attention.** The products stay at their 256-row build. A 1024-token block runs them in four
  256-row sub-blocks (measured a wash against one 1024-row build on the device, and it reads
  back less), every sub-block's scores before any values so a layer still switches context
  twice.

## Spec impact

- **New: OPEN-PREFILL-GEMM8.** The route, the `bfp16_dit` pack op, the `dit` context.
  - test: the C++ pack op equals `pack.pack_b` of the dequantised weight byte for byte
    (`pools_test.cpp`, a recipe test); the recipe emits the route for Qwen3-8B and not for a
    family whose shapes do not fit or that has no `prep`; the parser takes it and refuses a
    `bfp16_dit` op in a manifest below version 4.
  - manual: the accuracy gate (teacher-forced NLL over ~2k tokens of natural text within
    0.5 % relative of the q4_1 route, top-1 agreement reported; logits corr reported, not
    gated) and the 981-token profile; `oflm-test --llm` through `oflm serve`.
- **Modified: OPEN-PREFILL-MODE.** A dense set may carry the q4_1 route as `lean`, and a dense
  variant may run at a different t than `gemm_block` when every buffer around the GEMMs is
  sized for the larger.
- **Modified: OPEN-PREFILL-ATTN.** A dense block wider than the products' rows runs them per
  sub-block, scores before values across the block.
- **Manifest version 4** for `bfp16_dit`, as version 3 came with `bf16_gemm`.

## Out of scope

The SwiGLU epilogue and gathered A of `dit_gemm` (would remove the host SiLU pass), M = 512
blocks for short prompts, the 35B's and Qwen3.5's projections. Each is measured on its own
after this lands.
