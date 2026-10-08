# vae_ew: the VAE decoder's elementwise ops (GroupNorm, residual add, RGBA8)

This is one xclbin for FLUX.2 [klein]'s VAE decoder. Each op is an instruction stream
over it, selected by runtime loop counts (dit_ew's pattern). The kernels are in
`vew.cc`; the design and spec are documented in `vae_ew.py`.

| op | does | reads | writes |
|---|---|---|---|
| `gn_stats` | per-core partial GroupNorm sums (32 groups) | A | the layer's block in S |
| `gn_apply` | GroupNorm, optional SiLU | the block, then A | Y |
| `add` | residual add, optionally the next GroupNorm's sums | A, B | Y (in place is fine), Z → block |
| `rgba` | `clamp((x/2 + 1/2)·255)` of channels 0-2, alpha 255 | 4 channels per pixel | Y |

## Topology and layout

The topology is dit_ew's:
- 16 cores, 2 per column;
- per column, A and B are split between the pair, and Y and Z are joined;
- the unit is an element of 4096 bf16, always whole pixels × C;
- column c takes image rows [c·H/8, (c+1)·H/8).

Views are `{off, pitch, border, px_stride}`:
- `border` 1: dit_conv's zero-bordered NHWC.
- `border` 0: a plain [H·W, C] tensor.
- `px_stride` > C: the channels sit at the start of a wider row. The VAE attention's
  GroupNorm writes 512 of every 1024 columns; column 512 holds the GEMM's constant 1.

## How GroupNorm works

GroupNorm needs statistics over the whole tensor, so it takes two dispatches.
1. `gn_stats`, or `add` with `stats`, leaves each core's 32 × (Σx, Σx²) in the layer's
   parameter block.
2. `gn_apply` reads the whole block ahead of the data.

The block is 19 elements:
- a pad element;
- gamma at 0 and beta at 512, as bf16 (written by the host once);
- 16 partial-sum elements;
- a pad element.

The apply streams the block as p[−1], p[0], p[0], p[1], …, because a shim BD cannot
repeat with stride 0. So each core of a pair sees all of it, one element apart
(`vew_param`'s shift). The last partial element then turns the sums into a per-channel
`a = γ·rstd` and `b = β − mean·a`.

Arithmetic is dit_ew's:
- bf16 × bf16 MACs into fp32;
- float scales split into bf16 hi + lo;
- statistics are fp32 sums of exact products (x·1, x·x).

The GroupNorm output is rounded to bf16 before the SiLU, as diffusers' bf16 decode does.
The SiLU is dit_ew's tanh-based one, with ~1e-2 rel_fro against exact.

## Results (2026-09-26)

Test command: `make_test.py --test gn|addgn|rgba`, via `C:\dev\fa-work\vew_test.sh`.
Times are best of 3.

| test | rel_fro | time |
|---|---:|---:|
| GroupNorm 64×128×128 | 8.5e-5 | 0.19 + 0.28 ms |
| GroupNorm+SiLU 64×64×512, offset inputs | 1.1e-2 | 0.20 + 0.39 ms |
| add+stats then GroupNorm+SiLU 64×128×256 | add bit-exact, 1.0e-2 | 0.34 + 0.36 ms |
| GroupNorm into a plain [T, 512] view | 9.7e-5 | |
| rgba 64×256 | exact | 0.44 ms |
| GroupNorm+SiLU 512²×256 | 1.0e-2 | stats 2.7 ms (50 GB/s) + apply 4.8 ms (57 GB/s) |
| add+stats then GroupNorm+SiLU 1024²×128 | add bit-exact, 1.1e-2 | 12.8 ms (63 GB/s) + 9.2 ms (58 GB/s) |

These run at the DDR limit.

## Trap: rounding mode

Set the rounding mode in the first kernel of every dispatch.
- The core starts in floor rounding, and `vew_param` converts to bf16 before any data
  kernel runs.
- With the mode set only in the data kernels, the first dispatch after a context load was
  2× less accurate than every later one: 1.8e-4 vs 8.5e-5, from the same inputs.
- `vew_begin` sets it now.
