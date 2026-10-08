# dit-chain: whole FLUX.2 [klein] 4B blocks and its text encoder on the NPU

Phase 4 of `.claude/plans/image-diffusion-npu-only-plan.md`: before a 25-block engine can
hide them, catch layout and glue bugs (strides, concat offsets, head order, modulation
offsets, padding) by running whole blocks from captured inputs.

- `chain_test.py`: one double block and one or more single blocks. Every op on the NPU
  (dit_gemm with its SwiGLU epilogue, dit_ew, dit_fa) in the buffer layout
  `open_kernels/export_dit_kernels.py` encodes.
- `chain_test_te.py`: the text encoder, Qwen3-4B layers 1-27, 512 tokens, in the same
  three kernel sets.
- `safetensors_np.py`: reads (bf16) weights straight from the model's safetensors.
- `../../open_kernels/harness/npu_host.py`: the pyxrt host (hardware contexts per kernel
  set, sub-buffers of a root allocation, blocking waits).

- `chain_test_vae.py`: the VAE decoder. It runs the DiT's packed latents to RGBA8 in
  five kernel sets (conv, conv1, vew, gemm, fa), following
  `open_kernels/vae_decoder.py`'s schedule (102 dispatches).
- `generate.py`: a whole image, prompt to PNG, every op on the NPU:
  - it follows `open_kernels/klein_pipeline.py`'s schedule (1050 dispatches);
  - `--study` injects the quality study's prompts and noise;
  - `--profile` times every op;
  - `--pack-only` packs the weights (~8 GB, once).
- `export_bundle.py`: the same schedule as files for the native engine
  (`src/open_diffusion`). `klein_tokens.py` writes a prompt's token ids for it.

## Run

```
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_goldens.py --size 512      # and 1024
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_te_goldens.py
. C:\dev\mlir-aie\iron_env.ps1
python open_kernels\export_dit_kernels.py --resolutions 512,1024 --out C:\dev\klein-kernels
python utilities\dit-chain\chain_test.py --kernels C:\dev\klein-kernels --goldens C:\dev\ditref-out\goldens_512 --blocks dbl0,sgl0,sgl10
python utilities\dit-chain\chain_test_te.py --kernels C:\dev\klein-kernels --goldens C:\dev\ditref-out\goldens_te --fa <te_attn build for the prompt's length>
```

## Results (2026-09-26)

DiT blocks. Metric: rel_fro of the block's update (output − input), which the residual
would otherwise hide. The gate is the plan's: the NPU no further from diffusers than
1.25× the CPU emulation of its own arithmetic (`capture_goldens.py`'s `npu_*`).

| block | NPU vs emulation | NPU vs diffusers | emulation vs diffusers | bf16 output floor |
|---|---:|---:|---:|---:|
| double 0, 512² | 9.1e-3 | 9.8e-3 | 9.4e-3 | 2.6e-3 |
| single 0, 512² | 3.2e-2 | 3.6e-2 | 3.4e-2 | 2.0e-2 |
| single 10, 512² | 3.3e-2 | 3.6e-2 | 3.3e-2 | 1.9e-2 |
| double 0, 1024² | 1.1e-2 | 1.4e-2 | — | 2.6e-3 |
| single 0, 1024² | 3.4e-2 | 3.9e-2 | — | 1.6e-2 |
| single 10, 1024² | 2.9e-2 | 3.5e-2 | — | 1.6e-2 |

All PASS. Single blocks sit near 3e-2 because rounding the output to bf16 alone costs
~2e-2 of the update (the residual is large next to it), paid by both sides. 17
dispatches per double block, 6 per single. The SwiGLU epilogue reproduces dit_ew's
SwiGLU bit for bit (512² results identical before and after Phase 3).

Text encoder, 29 real tokens, 217 dispatches: real-token rows 5.9-7.3e-3 from the
emulation of the NPU arithmetic (emulation vs bf16: 1.2e-2); the zero-padded hidden
columns stay exactly 0 through 27 layers. PASS.

## The text encoder's padding rows

The prompt is right-padded to 512 and the DiT attends to all 512 rows, so the 483
padding rows' hidden states matter. They are hypersensitive to bfp16: 0.4-0.9 rel_fro on
the NPU against bf16, 0.2-0.7 in the emulation. Image study (`klein_quant_study.py`,
512², DiT in bf16, LPIPS vs bf16; noise floor 0.013):

| text encoder | LPIPS mean / max |
|---|---|
| NPU arithmetic, all rows | 0.092 / 0.24 |
| NPU real rows, bf16 padding rows | **0.017** / 0.039 |
| bf16 real rows, NPU padding rows | 0.086 / 0.21 |
| NPU arithmetic + power-of-two q/k smoothing | 0.085 / 0.13 |
| **plain fp32 text encoder** (no NPU arithmetic) | **0.053** / 0.084 |

The drift is entirely the padding rows, and it is not specific to the NPU: the padding
rows are chaotic under any numerical change -- a more accurate fp32 text encoder already
moves the images 0.053 from the bf16 run. The NPU's 0.09 is <2x that, so the bf16 pad
rows are not a target worth chasing. It is drift, not breakage: the worst prompt ("A
storefront sign that reads OPEN LATE") renders "OPEN LATE" where bf16 rendered only
"OPEN". **Decision: the text encoder runs on the NPU's normal arithmetic.**

Where the pad-row error comes from (layer 9, attention only): bfp16 Q/K dominate —
Qwen3's k_norm weights have outliers (layer 0: max 44 vs median 2). Balancing q/k per
RoPE pair by powers of two in the norm weights (exact, free) cuts real-row attention error
40x and pad-row error ~2x, but at image level it only reshuffles which prompts drift
(mean 0.092 -> 0.085), so it is not used.

## The VAE decoder (2026-09-26)

```
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\vae_study.py --size 512          # caches the 4-step latents
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_vae_goldens.py --size 512
python utilities\dit-chain\chain_test_vae.py --kernels C:\dev\klein-kernels --goldens C:\dev\ditref-out\goldens_vae_512 --runs 3
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_images.py C:\dev\ditref-out\goldens_vae_512
```

The chain test covers 8 prompts' latents and compares against diffusers' fp32 decode of
the same bf16 latents. At 512², every op runs on the NPU:

| stage (prompt 0) | rel_fro |
|---|---:|
| conv_in | 1.1e-2 |
| mid block | 5.4e-2 |
| up0 / up1 / up2 | 3.9e-2 / 4.4e-2 / 3.5e-2 |
| up3 | 1.8e-2 |
| conv_out | 1.8e-2 |

The images score PSNR 45.1 dB mean (min 43.8) and **LPIPS 0.0014 mean, 0.0017 max**,
10× under the DiT's noise floor. PASS.

The CPU emulation of the NPU arithmetic alone was 0.0002. The rest is the rank-128
attention (0.0004 alone) and the tanh-based SiLU.

A decode takes **~476 ms**:

| kernel set | time |
|---|---:|
| convs | 298 ms |
| vae_ew | 145 ms |
| 1×1 convs | 21 ms |
| attention (gemm + fa) | 10 ms |

Those times include the 71 kernel-set switches, which cost ~2.4 ms each (~170 ms). The
compute itself is ~300 ms. In the profile (`--profile`), 71 of the 102 ops run right after
a switch.

At 1024² (2 prompts, `goldens_vae_1024`) it also passes. The stages are within 1.2-6.2e-2,
the images score PSNR 44.3 dB and **LPIPS 0.0029 mean, 0.0032 max**, and a decode takes
**1.28-1.33 s**:

| kernel set | time |
|---|---:|
| convs | 0.81 s |
| vae_ew | 0.35 s |
| attention | 62 ms |
| 1×1 | 53 ms |

The largest ops are the upsample convs of up1 and up2 (101 and 86 ms) and the attention
(62 ms). The first decode after the sets are loaded takes 6.2 s (1.4 s at 512²): context
and instruction first-use. An engine warms up at load.

**Fixed 2026-09-27: the image's edge pixels.** Up blocks 2 and 3 narrow their channels
(512 → 256, 256 → 128) in their first resnet, and that resnet's GroupNorm output shared
a buffer with the narrower layout.
- Producers write the interior only, so each layout's interior landed on the other's
  zero border. The convs then read it as padding.
- The left, right and bottom edge pixels were off by 7-9 levels; the interior was ~1.
- Every decode depended on what the previous one left.

Now a zero-bordered buffer holds one channel count (`GI<i>`, `vae_decoder.plan`). Repeat
decodes are identical. At 512² the test gives PSNR **46.7 dB** and LPIPS **0.0011** (was
45.1 dB / 0.0014), and the late stages improved: up2 3.1e-2, up3 1.7e-2, conv_out
1.5e-2.

## The whole image (Phase 6, 2026-09-27)

```
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_pipeline_inputs.py --size 512
python utilities\dit-chain\generate.py --kernels C:\dev\klein-kernels --size 512 --study C:\dev\ditref-out\goldens_pipe_512 --out C:\dev\gen\full512
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_images.py C:\dev\gen\full512 --test "{:02d}.png" --ref "{:02d}.png" --ref-dir C:\dev\ditref-out\klein_512_s4\bf16
```

The run takes 8 study prompts with the study's fixed noise (512²). LPIPS against the bf16
CPU run:

| | LPIPS mean / max |
|---|---|
| everything on the NPU | 0.107 / 0.191 |
| DiT + VAE on the NPU, bf16 text embeddings (`--ctx-ref`) | 0.044 / 0.092 |
| (CPU emulation of the DiT's linears + attention) | (0.029 / 0.081) |
| (CPU emulation of the text encoder alone) | (0.092 / 0.241) |

- The images are coherent, and the text prompts render legibly. The drift is what the
  studies predict.
- The extra DiT drift is the conditioning GEMMs: timestep MLP, modulation and embedders.
  They share dit_gemm's bf16-accumulator arithmetic, which the study left in bf16, and
  the modulation vectors come out 2-3% from diffusers'.
  - Injecting diffusers' exact vectors takes the 7 non-chaotic prompts from 0.038 to
    0.028.
- `generate.py` and `src/open_diffusion` give pixel-identical images for the same inputs.
- The DiT's final latents are deterministic run to run.
