---
name: dit-ew
description: Build, verify and time dit_ew, the open XDNA2 kernel set for a diffusion transformer's elementwise and row ops (LayerNorm+modulate, residual+LayerNorm+modulate, q/k RMSNorm+RoPE, SwiGLU, Euler, SiLU) for FLUX.2 klein 4B, and write fast bf16/fp32 vector kernels on aie2p. Use when adding an op, a new DiT's norm/RoPE variant, the text encoder's RMSNorm/RoPE, or when an aie2p elementwise kernel runs far slower than its DMA.
---

# dit_ew: elementwise ops for the DiT family

Source: `open_kernels/designs/dit_ew/` (README has the numbers). Tests: `make_test.py`
(spec + inputs + fp64 reference per op), `compare.py`.

## What was learned getting here (don't re-derive it)

1. **aie2p has no native fp32 vector multiply.** Kernels written in `aie::vector<float,16>`
   ran 4-6× slower than their DMA. Use bf16 × bf16 → accfloat MACs on 32 lanes; apply a
   float scalar as bf16 hi + lo (two MACs), exact to fp32 rounding.
2. **Hardware tanh and exp2 are approximations** (`utilities/aie-probes/`): exp2 is
   Mitchell's linear mantissa (+3.8% mean), tanh 1.4-3.5% rms below |x| = 2. bf16
   `inv`/`invsqrt` are bf16-accurate but scalar loops. SiLU as `0.5x(1 + tanh(x/2))`
   keeps tanh's error under ~0.5%.
3. **Memtile channel budget decides the core count**: 2 inputs + 2 outputs per core with
   split/join through the memtile is 6 S2MM + 6 MM2S per column only at 2 cores per
   column (16 cores). The ops are DDR-bound, so that is fine.
4. **A shim BD's d0 is at most 1023 words**: a 3072-bf16 row goes as `[.., 3, 1024]`.
   (dit_fa: the outermost/repeat dim wraps silently above 64.)
5. **No stride-0 BD dimension** ("Stride must be a positive integer"). To send each
   parameter vector to both cores of a split pair, stream `p[-1], p[0], p[0], …, p[n]`
   (one 4-D fill, strides `[EL, EL, 1024, 1]`) and let each core keep its share.
6. **Runtime op selection**: one `range_` loop per op with an RTP trip count; zero for
   the ops not requested. All ops share one static configuration.
7. **RoPE in-core**: klein's position ids are (0,0,0,l) for text and (0,h,w,0) for image
   tokens, theta 2000, 4 axes × 32 dims, interleaved pairs. A 64-position fp32 table
   covers h/w; text positions use angle addition with a 64-multiple table. No per-token
   tables are streamed.
8. **Timing traps**: never time NPU builds in parallel (they overlap on the array), and
   check that a `sed` edit of a generated cfg matched (Windows paths use backslashes).
   `DE_NOP=1` gives a stream's DMA floor.

9. **Text-encoder modes**: `"norm": "rms"` (+ `"unit_gate": 1`) and `"rope": "qwen"` with
   head roles (`b_q_heads`, `b_k_heads`) cover Qwen3; hidden sizes below 3072 run with
   `"W"` and zero padding columns. Keep the qk helpers `noinline` (16 KB program memory).

## Build / test one op

```
python open_kernels\designs\dit_ew\make_test.py --op res_ln_mod --out <t>
$env:DE_SPEC = Get-Content <t>\spec.json; . C:\dev\mlir-aie\iron_env.ps1
python open_kernels\build_design.py open_kernels\designs\dit_ew\dit_ew.py <b>
python open_kernels\designs\dit_ew\make_test.py --op res_ln_mod --out <t> --build <b>
open_kernels\harness\out\run_kernel.exe <t>\run.cfg
python open_kernels\designs\dit_ew\compare.py <t>
```

Expected: PASS at rel_fro 1.6-2.8e-3 (norm ops, RoPE, Euler; bf16 floor 1.7e-3) and
3.5-5.4e-3 (SiLU, SwiGLU).

## Adding an op

Add a kernel in `ew.cc`, a loop + RTP count slot in `dit_ew.py` (keep the fifo set: A, B
in; Y, Z out; params via A), a case in `make_test.py` with the diffusers formula as the
reference. Keep scalar×vector math in hi/lo MACs.
