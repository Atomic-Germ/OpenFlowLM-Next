# dit_ew: the elementwise and row-wise ops of a FLUX.2 [klein] denoising step

`dit_ew.py` (IRON) + `ew.cc` (kernels). One static xclbin; every op/shape is an
instruction stream over it (checked: the six op builds below are identical modulo UUID,
`export_gemm_rtp.xclbin_identical_mod_uuid`).

| op | math (diffusers `transformer_flux2.py`) | streams |
|---|---|---|
| `ln_mod` | `LN(x) * (1 + scale) + shift` (norm1/norm2/norm_out) | A → Y |
| `res_ln_mod` | `x' = x + gate * y`, then `LN(x') * (1 + scale) + shift` | A, B → Y (z), Z (x') |
| `qk` | per-head RMSNorm (`norm_q`/`norm_k` weights), then 4-axis interleaved RoPE, q and k at once | A, B → Y, Z |
| `swiglu` | `silu(g) * u` (`Flux2SwiGLU`) | A, B → Y |
| `euler` | `x + dt * v` (flow-match Euler step) | A, B → Y |
| `silu` | `silu(x)` (timestep MLP, modulation input) | A → Y |

`res_ln_mod` exists because klein's modulation is shared by every block: the residual
update after attention or the MLP is always followed by the next LayerNorm+modulate, so
the pair is one pass over the residual stream.

## Status (2026-09-26, HX 370, turbo), 4608 rows (klein 1024², joint sequence)

| op | time | DMA floor (no compute) | traffic | rel_fro vs fp64 (bf16 floor) |
|---|---:|---:|---:|---:|
| ln_mod | 1.54 ms | 1.17 ms | 57 MB | 2.8e-3 (1.7e-3) |
| res_ln_mod | 2.00 ms | 2.14 ms | 113 MB | Y 2.8e-3, Z 1.6e-3 (1.7e-3) |
| qk | 3.2 ms | 2.28 ms | 113 MB | 2.6e-3 (1.7e-3) |
| swiglu (9216 wide) | 4.2 ms | 4.42 ms | 255 MB | 5.4e-3 (1.7e-3) |
| euler (4096 × 128) | 0.17 ms | — | — | 1.7e-3 (1.7e-3) |
| silu (16 rows) | 0.13 ms | — | — | 3.5e-3 (1.6e-3) |

res_ln_mod and swiglu run at the DDR limit (~55-60 GB/s); ln_mod and qk are ~30-40%
over theirs. The norm ops' error above the floor is the extra bf16 rounding of the norm
output before the modulation/weight, which diffusers' bf16 pipeline also does; SiLU's
is the hardware tanh (below). About 0.25 s per 1024² step in total at these speeds
(5 double + 20 single blocks); the dit_gemm epilogue fusions of the plan (SwiGLU, then
qk norm+RoPE) remove the two largest.

## Topology

- 16 cores, 2 per column (rows 2-3). Per column, A and B come in through the memtile and
  split between the pair; Y and Z join there on the way out. That is 6 memtile S2MM and
  6 MM2S — the budget, hence 2 cores per column. The ops are DDR-bound; 16 is plenty.
- The core program has one loop per op with a runtime trip count (RTP): zero for the ops
  not asked for. One configuration serves every op.
- The work unit is a row element of 3072 bf16 (klein's hidden size; a q/k row is 24
  heads). A stream reads a *view* of a buffer — T tokens, each E consecutive elements of
  3072 at column `off`, row stride `ld` — so the fused QKV / ff_in buffers are read in
  place. Column c takes tokens `[c*T/8, (c+1)*T/8)`; within a column the elements
  alternate between the two cores. The shim BD is `[T/8, E*3, 1024]` (d0 ≤ 1023 words).
- Euler reads latents as `tile24` views (24 tokens × 128 channels per element), so the
  velocity can come straight out of an N-padded proj_out buffer.
- Parameter vectors ride stream A ahead of the rows. A shim BD cannot stride by 0, so one
  fill streams `p[-1], p[0], p[0], p[1], …, p[n]` and each core of the pair keeps its
  share (the run needs one readable vector on each side).
- RoPE is generated in-core from 9 KB of fp32 tables in the parameter block: FINE
  (positions 0-63) for klein's h/w axes, FINE + COARSE (multiples of 64) by angle
  addition for the text index axis. Nothing per token is streamed.

## Arithmetic

aie2p's fp32 vector multiply is emulated: the first version of these kernels, written in
fp32 16-lane vectors, ran 4-6× slower than its DMA (ln_mod 6.6-9.4 ms). The kernels now
use bf16 × bf16 → fp32-accumulator MACs on 32 lanes; a float scalar that multiplies a
vector (1/std, dt, cos/sin) is split into bf16 hi + lo and applied as two MACs, which is
exact to fp32 rounding.

SiLU is `0.5x(1 + tanh(x/2))`. aie2p's vector tanh is an approximation (1.4-3.5% rms
below |x| = 2, `utilities/aie-probes/vecmath_probe.py`); the leading 1 keeps its error
under ~0.5% of the result. bf16 `inv`/`invsqrt` are bf16-accurate but scalar loops, so
they appear only per row.

## Build and test

```
python open_kernels\designs\dit_ew\make_test.py --op qk --out <t>            # spec.json + inputs + reference
$env:DE_SPEC = Get-Content <t>\spec.json
python open_kernels\build_design.py open_kernels\designs\dit_ew\dit_ew.py <build>
python open_kernels\designs\dit_ew\make_test.py --op qk --out <t> --build <build>
open_kernels\harness\out\run_kernel.exe <t>\run.cfg
python open_kernels\designs\dit_ew\compare.py <t>
```

`DE_NOP=1` builds a variant whose kernels return at once: the DMA floor of a stream.

## Text-encoder modes (Qwen3-4B, added with the Phase 4 chain)

| spec | op |
|---|---|
| `"norm": "rms"` on `ln_mod` / `res_ln_mod` | RMSNorm × weight (HF Qwen3RMSNorm's rounding: normalised value to bf16, then × weight); with `"unit_gate": 1` the residual is a plain add |
| `"rope": "qwen"`, `"b_q_heads": 8`, `"b_k_heads": 8` on `qk` | Qwen3's fused q\|k\|v row [512, 6144] as two elements: A = q heads 0-23, B = q heads 24-31, k heads 0-7, v heads 0-7 copied through; rotate-half RoPE, θ = 1e6, positions 0-511 by two angle additions over three 8-entry fp32 tables |
| `"W": 2560` | the hidden size inside the 3072 element; the padding columns are written as zeros |

All pass standalone at the bf16 floor (`make_test.py --op rms / res_rms / qk_qwen`), and
the 27-layer text encoder runs on them (`utilities/dit-chain/`). The qk helpers are
`noinline`: inlined three times each they overflow the core's 16 KB program memory.

The SwiGLU op is no longer used by klein: dit_gemm's epilogue does it (Phase 3).

## Not done

- VAE ops (GroupNorm stats/apply+SiLU, upsample, to-RGB8) come with `dit_conv`.
