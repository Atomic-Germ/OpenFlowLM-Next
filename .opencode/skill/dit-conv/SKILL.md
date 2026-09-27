---
name: dit-conv
description: Build, verify, time and extend the open XDNA2 convolution kernel (dit_conv, implicit GEMM with the 3x3 window read out of the memtile through explicit IRON TileDma, 11.7-17 TFLOPS) and the FLUX.2 klein VAE decoder schedule that uses it with vae_ew, dit_gemm and dit_fa. Use when rebuilding the VAE kernel sets, adding a VAE / conv shape or another conv model (TAEF2, other decoders), debugging dit_conv hangs, BD-pool or channel errors, or when an IRON ObjectFifo cannot express a DMA pattern.
---

# dit_conv and the VAE decoder

Sources:
- `open_kernels/designs/dit_conv/` (README has the numbers and the dataflow)
- `open_kernels/designs/vae_ew/` (the GroupNorm / add / RGBA ops)
- `open_kernels/vae_decoder.py`: the weight packing and the whole schedule. It is the
  single source of truth for both the exporter and the runner.

Exporter: `open_kernels/export_dit_kernels.py` builds `conv/`, `conv1/` and `vew/`, and
adds the VAE's attention streams to the gemm and fa sets. `--no-vae` skips all of it.

Design note: `.claude/plans/image-diffusion-phase5-vae.md`.

## What was learned getting here (don't re-derive it)

1. **VAE numerics don't matter.**
   - `utilities/dit-ref/vae_study.py`: the NPU's bf16/bfp16 arithmetic costs LPIPS 0.0002
     against fp32.
   - The pipeline decodes with the bf16 VAE despite `force_upcast`.
   - Judge VAE changes on speed.
2. **dit_gemm streams can't do the VAE's convs.** Its N % 1024 means ≤ 50% of the array
   at Cout ≤ 512, and shim-level windowing reads activations 9× from DDR.
3. **An ObjectFifo link's MM2S length is always its L2 object's size**
   (`AIEObjectFifoStatefulTransform.cpp:1435-1510`; distribute lengths come from
   offset differences).
   - So no ObjectFifo can re-read an L2 buffer (windowing).
   - Use IRON's `TileDma`/`Bd`/`Lock`/`Flow` (`aie.iron`, this mlir-aie has them). They
     mix with ObjectFifos in one `Program`: the lowering appends to the same
     `aie.memtile_dma`/`aie.mem` regions and skips channels already claimed, including
     a `Flow`'s `shim_symbol` allocation.
   - Drive an explicit shim channel with `shim_dma_single_bd_task(symbol, x.op, ...)` +
     `dma_start_task`, and append `(dma_free_task, [task])` to the TaskGroup's
     `_actions`.
4. **The core's A S2MM wants each 32×64 quarter column-block-major** (dit_gemm's
   a_l2l1 pattern).
   - A row-major band would need a 6-D read, so store the band **chunk-major**
     `[8 ch-block][rows][px][8]` via the memtile S2MM write pattern.
   - Give each ky its own BD: `[kx 3][quarter 4][ch-block 8][256]` at row r+ky.
   - Lock value 12 = 4 readers × 3 BDs.
5. **aie2p memtile BD pools**: channels 0/2/4 share BDs 0-23, channels 1/3/5 share 24-47.
   - A join's output takes depth × parts BDs (8).
   - Count with `open_kernels/emit_mlir.py --lower` (it undercounts explicit
     chains), then spread explicit channels (`READER_CH = [0, 2, 4, 1]`).
6. **Grids narrower than 128 px** (the 512 px image's 64² latent, the 32² packed
   latent) compute garbage parts that are drained past the row's right border. Output
   pitch must be (128/W)(Wo+2).
7. **Upsample + conv = 4 output phases on the source grid**, with the taps summed
   (`conv_pack.phase_weights`). No upsampled tensor exists.
   - The same "up" mode with 1×1 does the pipeline's unpatchify + BN +
     post_quant_conv (`vae_decoder.latent_in_weights`).
8. **Bias rides the B stream** as a raw-bf16 first object per tile (`conv.cc`), so
   no bias pass.
9. **Set the rounding mode in the first kernel of every dispatch** (vae_ew's
   `vew_begin`). The core starts in floor, and a parameter kernel that converts to
   bf16 before any data kernel made the first dispatch after a context load 2× less
   accurate.
10. **The VAE's single d=512 attention head runs on unchanged dit_fa.**
    - The score form `[x,1] Ma xᵀ` is factored at rank 128 (plain SVD, LPIPS 0.0004;
      data-weighted 0.0000, `utilities/dit-ref/vae_attn_rank.py`).
    - It runs as 4 heads that share q′ and k′, over V's 128-column slices.
    - q′ carries ½, because dit_fa scales by 1/√128.
    - W_o and the V/O biases fold into v″, and GEMM biases come through a constant-1
      column in QIN.

## Verify

```
# per kernel (C:\dev\fa-work\conv_test.sh / vew_test.sh batch: spec -> build -> run -> check)
python open_kernels/designs/dit_conv/make_test.py --H 128 --W 128 --Cin 512 --Cout 512 --out <t> [--build <b>]
python open_kernels/designs/vae_ew/make_test.py --test gn|addgn|rgba --C 128 --H 64 --W 128 --silu --out <t> [--build <b>]
# whole decoder
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_vae_goldens.py --size 512
python utilities\dit-chain\chain_test_vae.py --kernels C:\dev\klein-kernels --goldens C:\dev\ditref-out\goldens_vae_512
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_images.py C:\dev\ditref-out\goldens_vae_512
```
