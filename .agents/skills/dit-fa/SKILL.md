---
name: dit-fa
description: Build, verify, time and model the open XDNA2 flash-attention kernel for diffusion transformers (dit_fa, head dim 128, 8.6 TFLOPS on FLUX.2 klein 4B at 1024²) and its text encoder (causal, GQA, padded prompt). Use when rebuilding the klein attention set, adding a resolution or another d=128 model, debugging dit_fa hangs or wrong output, deciding on the exp2 accuracy fix, or emulating NPU attention numerics on the CPU.
---

# dit_fa: attention for the DiT family

Source: `open_kernels/designs/dit_fa/` (README has the numbers and the topology).
Exporter: `open_kernels/export_dit_kernels.py` builds it into `<set>/fa/` next to the
dit_gemm set (`--no-fa` skips, `--no-gemm` builds attention only, `--fa-exp-fix`).
CPU model: `utilities/dit-ref/fa_emul.py`. exp2 probe: `utilities/aie-probes/exp2_probe.py`.

## What was learned getting here (don't re-derive it)

1. **whisper_fa's cascade topology is latency-bound**: ~1.9 TFLOPS at its own shape
   (20 × 1536² × 64). dit_fa drops the cascade: 2 groups of 16 cores, one head per group,
   32 query rows per core, every core walks all keys; K and V are broadcast per group.
   24 × 4608² × 128: 40.6 ms on the first version, 30.5 ms (8.6 TFLOPS) after items 7-8.
2. **d = 128 does not fit L1 unchunked.** The core splits d into two 64-wide chunks for
   the mmul; Q is captured as two A tiles; the output accumulator is the acquired output
   element itself (no second 8 KB buffer).
3. **A shim BD's outermost dimension wraps silently above 64** (a 72-chunk fill re-read
   chunks 0-7: right size, garbage data, no hang). Splitting one stream over several
   fills per pass hung. Use 2-D/3-D fills with the large count in a middle dimension, and
   let the memtile do layout work (`_block_dims(64, 128)` emits both d-chunks in mmul
   order).
4. **A join's `dims_to_stream` applies per joined part.**
5. **`-DDC=` breaks aie_api** (`AIE_RegFile::DC`): kernel knobs are `FA_*`.
6. **aie2p's `aie::exp2<bfloat16>` is Mitchell's approximation**: `2^n * (1 + f)`, f the
   input fraction truncated to 7 bits. +3.8% mean, +6.1% at half-integers, 0 at
   integers; for |x| >= 2 the input is floored to the bf16 grid with the missing bits read
   as ones. `fa_emul.hw_exp2` is bit-exact. AMD's own aie2p softmax uses it too.
   `FA_EXP_FIX` corrects it with a quadratic in the returned mantissa (rel_fro 2.58e-2 →
   1.88e-2 on random data) for +9% kernel time; off by default (see Quality).
7. **Lazy rescale is faster and more accurate**: move the reference max only when a row's
   chunk max exceeds it by 8 (log2 units); P ≤ 256 is harmless in bf16. Eager rescaling
   re-rounds O to bf16 every chunk (2.84e-2 vs 2.58e-2).
8. **Row sums on the MAC array** (P @ ones into a replicated fp32 l) replace 32 scalar
   reductions and normalise by the same bfp16 P that P·V used.
9. **Stack**: the exp-fix build has a 3 KB `softmax_step` frame; at a 3 KB stack the array
   hung. The design uses 4 KB. Read frames from `llvm-objdump -d` (`st ..., [sp, #-N]`).
10. **Masked scores are bf16 `lowest`**, not −inf (the exp argument is built in fp32 as
    `S*c - m*c`).
11. **No in-core cycle counter under Peano.** `aie::tile::current().cycles()` compiles
    but `get_cycles()` is defined nowhere (link error); reading the core timer register
    with `__builtin_aie2p_read_tm((void*)0x340F8)` hangs the array. Time pieces with
    ablation builds (`DF_EXPT`) on a quiet machine instead.
12. **XRT's `run.wait()` sleeps on Windows**: 119 dispatches over 6 s of wall time cost
    ≤ 30 ms of host CPU. A blocking wait is safe for the no-CPU-spin rule.

13. **Layouts**: `DF_QKV_LD`/`DF_K_COL`/`DF_V_COL` read Q/K/V in place from a fused buffer;
    `DF_O_INTERLEAVE` writes O 64-of-128 for dit_gemm's gathered A. Wide buffers: K/V go
    in row blocks small enough that the block stride stays under 2^20 words.
14. **The text encoder's padding rows are the numerics risk**, not the DiT: see
    `utilities/dit-chain/README.md` (bfp16 Q/K with Qwen3's k_norm outliers).

## Build / test

```
. C:\dev\mlir-aie\iron_env.ps1
python open_kernels\export_dit_kernels.py --no-gemm            # -> src\xclbins\<family>\open_kernels\fa
xrt-smi configure --pmode turbo
python open_kernels\designs\dit_fa\make_test.py --kernels <set>\fa --stream r1024_attn --out <t>
open_kernels\harness\out\run_kernel.exe <t>\run.cfg
python open_kernels\designs\dit_fa\compare.py <t>
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\fa_emul.py --check <t>
```

Single shapes: `DF_L`, `DF_HEADS`, `DF_KV_HEADS`, `DF_CAUSAL`, `DF_VALID_LEN`,
`DF_EXP_FIX` with `open_kernels\build_design.py open_kernels\designs\dit_fa\dit_fa.py <out>`
(bash: set them in the environment of `powershell.exe`, not with backticks).
`DF_EXPT=noqk,nopv,nosm` builds ablation variants that skip that compute (timing only).

Expected: PASS at rel_fro ~2.6e-2 (random N(0,1) data), row cosine > 0.999; `fa_emul
--check` shows the model within ~7e-3 of the hardware. A model/hardware gap near the full
error size means the kernel's arithmetic changed and the model did not.

| symptom | meaning |
|---|---|
| state 8 (timeout) | fill/acquire count mismatch, a split fill, or a stack overflow |
| rel_fro ~1-5, no hang | a fill whose outer dim > 64, or a layout transform |
| rel_fro a few %, cos ~0.99 | exp2 / rounding: compare with fa_emul stages |

## Quality

`utilities/dit-ref/klein_quant_study.py` variants `attn-fa`, `attn-fa-exact` and
`bfp16-bf16acc+attn-fa` (the whole NPU DiT arithmetic) route diffusers' Flux2 attention
through `fa_emul.dit_fa_attention`. Results in the design README. At 512²: hardware exp LPIPS 0.022,
exact exp 0.016, whole NPU DiT 0.029 (noise floor 0.013; w8a8 0.051) — drift, not
breakage, so the fix is off by default.

## Open

- `te_attn`'s `valid_len` (the prompt length) is baked into the instruction stream as an
  RTP write; the engine must patch it per prompt.
- VAE attention (d = 512, one head, 16384 tokens) needs its own Q/output split.
