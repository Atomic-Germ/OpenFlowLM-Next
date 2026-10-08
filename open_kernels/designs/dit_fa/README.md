# dit_fa: flash attention at head dim 128 (FLUX.2 [klein] 4B and its text encoder)

`dit_fa.py` (IRON) + `fa_dit.cc` (kernels). One xclbin; every shape is an instruction
stream over it (checked: the 512-token, 1536-token, 4608-token and causal-GQA builds are
identical modulo UUID, `export_gemm_rtp.xclbin_identical_mod_uuid`).

    O[t, h*128:(h+1)*128] = softmax(Q_h K_kv(h)^T / sqrt(128)) V_kv(h)

Q, K, V, O are bf16 and token-major (`[tokens, ld]`, head h at columns `col + h*128`), the
layout the QKV GEMM writes and the out-projection reads. Q/K/V can be three views of one
fused buffer. Options per stream: `kv_heads` (GQA), `causal`, `valid_len` (keys at or past
it are masked; the text encoder's padded prompt).

## Status (2026-09-26, HX 370, turbo, quiet machine)

| build | 24 heads x 4608² x 128 (klein 1024², joint) | TFLOPS | rel_fro vs fp64 | min row cos |
|---|---:|---:|---:|---:|
| first version (eager rescale) | 40.6 ms | 6.4 | 2.9e-2 | 0.9991 |
| **current** (lazy rescale, row sums on the MAC array, fp32 exp argument) | **30.5 ms** | **8.6** | 2.6e-2 | 0.9993 |
| current + `FA_EXP_FIX` | 33.2 ms | 7.9 | 1.9e-2 | 0.9997 |

Random N(0,1) Q/K/V, median of 11 warm runs. The text encoder's mode (32/8 heads x 512²,
causal, valid_len 77) passes at the same error. The current build is ~0.76 s of attention
per 1024² denoising step (25 blocks); the plan's gate was 6 TFLOPS.

A hardware-context change between dit_gemm and dit_fa costs 2.2-2.5 ms each way
(`xclbin` x2 in one run_kernel cfg, alternating dispatches vs solo).

## Topology, and why it is not whisper_fa's

- 32 cores = 2 groups of 16 (columns 0-3, 4-7). A group works on one head; each core
  owns 32 query rows (a pass = 512 rows) and walks **all** keys itself. No cascade, no
  merge.
- K reaches a group through one memtile (fifo A), V through another (fifo B), each
  broadcast to all 16 cores. The pass's Q tiles ride the same broadcasts ahead of the
  keys; each core keeps its own rows (whisper_fa's selective capture).
- L3 → memtile moves whole `[64, 128]` row blocks; the memtile emits each as its two
  `[64, 64]` head-dim chunks, already in mmul block order. The core splits d = 128 into
  two 64-wide chunks for the mmul calls.
- Output: the core accumulates straight into its acquired output element (no extra
  L1); a column's 4 cores join at their memtile, which streams the 128-row block out
  row-major.
- L1 per core ≈ 59 KB of 64: A/B fifos 2 × 2 × 8 KB, output 8 KB, Q 8 KB, scores 4 KB,
  running sum 1 KB, .bss 1.7 KB, stack 4 KB.

whisper_fa (the cascade port of MLIR-AIR's `kernel_fusion_based`) measured **~1.9 TFLOPS**
at its own shape (20 × 1536² × 64, 6.3 ms): 60 serialized 6-chunk iterations with a full
drain between each, and K/V re-streamed once per 256 query rows. Here a pass is 72 chunks
at 4608 tokens, two passes are in flight, and K/V cross DDR once per 512 query rows.
Upstream MLIR-AIR's own newer designs (`attn_npu2_headspatial`, `temporal_causal`) drop
the cascade for the same reasons.

Where the time goes (ablation at 4608 × 2 heads, 3.38 ms, first version): no compute at
all 1.17 ms (the DMA floor); without the softmax update −1.35 ms; without P·V −0.56 ms;
without Q·Kᵀ no change. The softmax update was ~40% — hence the lazy rescale and the
row sums on the MAC array.

## Accuracy

`utilities/dit-ref/fa_emul.py` models this kernel's arithmetic and matches hardware to
~7e-3 (errors ~2-3e-2 vs fp64). Attribution (current kernel, random inputs):

| change | rel_fro |
|---|---:|
| as built | 2.58e-2 |
| exp2 exactly rounded | 1.85e-2 |
| `FA_EXP_FIX` | 1.88e-2 |
| eager rescale (first version) | 2.84e-2 |
| Q/K, P/V, accumulator in higher precision | ~2.4e-2 each |

**The aie2p hardware exp2 is not bf16-accurate.** `aie::exp2<bfloat16>` returns
`2^n * (1 + f)` with f the fraction of the input truncated to 7 bits — a linear
mantissa (Mitchell's approximation): exact at integers, +6.1% at half-integers, +3.8% mean
(`utilities/aie-probes/exp2_probe.py` dumps it; `fa_emul.hw_exp2` is bit-exact). The mean
cancels in the softmax normalisation; the variation does not. AMD's own aie2p softmax
kernel uses the same instruction. `FA_EXP_FIX` (build flag, `DF_EXP_FIX=1`) multiplies by
a quadratic in the returned mantissa; it costs a 3 KB stack frame and +9% kernel time
(30.5 -> 33.2 ms).

**Decision (2026-09-26): shipped without it.** Image study (`utilities/dit-ref/
klein_quant_study.py`, the first kernel's arithmetic, `C:\dev\ditref-out\klein_512_s4`):

| variant (512², 4 steps, 8 prompts) | LPIPS vs bf16 mean / max | PSNR |
|---|---|---:|
| fp32 (noise floor) | 0.0129 / 0.031 | 33.4 dB |
| dit_gemm arithmetic | 0.0150 / 0.031 | 31.7 dB |
| attention, exact exp (`attn-fa-exact`) | 0.0164 / 0.033 | 30.7 dB |
| attention, hardware exp (`attn-fa`) | 0.0220 / 0.067 | 30.2 dB |
| GEMM + attention, hardware exp (the whole NPU DiT) | 0.0288 / 0.081 | 28.3 dB |
| (w8a8 GEMM, for scale: drift, not broken) | 0.0512 / 0.132 | 25.7 dB |

Text stays correct and legible in every variant; the hardware exp moves small details
(sign framing, rug pattern), the worst prompt being the neon sign. That is drift, not
breakage, under the project's "maximum speed without breaking quality" rule. Build with
`--fa-exp-fix` / `DF_EXP_FIX=1` if drift ever matters.

## Build and test

```
. C:\dev\mlir-aie\iron_env.ps1
$env:DF_L=4608; $env:DF_HEADS=24          # DF_KV_HEADS, DF_CAUSAL, DF_VALID_LEN, DF_EXP_FIX
python open_kernels\build_design.py open_kernels\designs\dit_fa\dit_fa.py <out>
xrt-smi configure --pmode turbo
python open_kernels\designs\dit_fa\make_test.py --xclbin <out>\final.xclbin --insts <out>\insts.bin --out <t> --L 4608 --heads 24
open_kernels\harness\out\run_kernel.exe <t>\run.cfg
python open_kernels\designs\dit_fa\compare.py <t>
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\fa_emul.py --check <t>
```

Shape constraints: tokens % 512, keys % 64, heads % 2.

## Layouts (for the chain tests / engine)

Build-time (instruction stream) options, `DF_*` env for `build_design.py`, `layout` in
`export_dit_kernels.py`'s streams:

- `DF_QKV_LD`, `DF_K_COL`, `DF_V_COL`: Q/K/V read in place out of one fused buffer (klein's
  double-block QKV [T, 9216], single-block FU [T, 33792], the text encoder's [512, 6144]).
- `DF_O_LD`, `DF_O_COL`, `DF_O_INTERLEAVE`: the output's row stride and column, and with
  interleave each head's 128 dims written as two 64-column halves 128 apart — the
  64-of-128 layout dit_gemm's gathered A reads, so a single block's out-projection reads
  [attention | SwiGLU] straight out of FU.
- K/V fills go as `[lk/rs, rs, 128]` row blocks, `rs` rows small enough that `rs * ld`
  stays under a BD stride's 2^20 words (64 rows of a 33792-wide buffer did not).

## Traps hit getting here

1. **A shim BD's outermost dimension wraps silently above 64.** A `[72, 2, 64, 64]` K
   fill read chunks 0-7 nine times: right byte count, no hang, garbage output. Splitting
   the stream into several fills per pass hung instead. Fix: 2-D/3-D fills of whole row
   blocks (`[n_chunks, 64, 128]` has its large count in a middle dimension) and the
   head-dim split done by the memtile.
2. **A join's `dims_to_stream` applies per joined part**, not to the joined buffer.
3. **`-DDC=...` breaks aie_api** (`AIE_RegFile::DC`); the kernel's knobs are `FA_*`.
4. **Stack**: `softmax_step` with `FA_EXP_FIX` has a 3 KB frame; at a 3 KB stack the
   array hung. Check `st ..., [sp, #-N]` in `llvm-objdump -d` when a kernel grows.
5. Masked scores are bf16 `lowest`, not −inf: the exp argument is `S*c - m*c` in fp32,
   and −inf would stay −inf through a path that expects finite values.

## Not done

- Speed pass: exp-fix cost (a LUT via `aie::parallel_lookup` instead of the quadratic),
  the text encoder's per-element ragged mask, K/V in bfp16 from their producer.
- The text encoder's `valid_len` is per prompt: it is an RTP value in the instruction
  stream, so the engine has to patch it (or the stream is rebuilt per length).
- The VAE's d = 512 attention (one head, 16384 tokens) needs a different Q/output split.
- Export: `export_dit_kernels.py` does not build dit_fa streams yet.
