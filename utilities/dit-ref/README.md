# dit-ref: CPU reference and GEMM-arithmetic study for FLUX.2 [klein] 4B

`klein_quant_study.py` runs the diffusers pipeline on the CPU. It swaps every linear
layer in the transformer's double- and single-stream blocks (100 layers, 3.68B of the
model's parameters) for a numerics-faithful emulation of one NPU GEMM datapath, then
scores the images against the bf16 run. The docstring lists the variants.

Setup:

```
python -m venv --system-site-packages C:\dev\ditref-venv
C:\dev\ditref-venv\Scripts\python.exe -m pip install "diffusers>=0.37.0" lpips
hf download black-forest-labs/FLUX.2-klein-4B --exclude flux-2-klein-4b.safetensors
```

The exclude skips a redundant 7.75 GB single-file copy; the diffusers layout is ~15 GB.

Run:

```
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\klein_quant_study.py --size 512
```

Output goes to `C:\dev\ditref-out\klein_<size>_s<steps>\`:
- one directory per variant;
- `report.json`;
- `grid.png` (rows = prompts, columns = variants).

Finished images are skipped, so a run can be resumed.

## Result (2026-09-25, 512², 4 steps, 8 prompts, fixed noise)

| variant | what it emulates | LPIPS vs bf16 (mean / max) | PSNR |
|---|---|---|---|
| fp32 | noise floor of the metric | 0.0129 / 0.0305 | 33.4 dB |
| bfp16-emul | `gemm_pretiled --emulate-bfp16` | 0.0132 / 0.0371 | 32.9 dB |
| **bfp16-bf16acc** | **`atbs` (and `wam`): bf16 x bfp16, accumulator re-rounded to bf16 every 64 of K** | **0.0150 / 0.0305** | 31.7 dB |
| w8a8 | `gemm_pretiled --int8`, per-token activations, per-channel weights | 0.0512 / 0.1321 | 25.7 dB |

- **bfp16 either way is at the noise floor.** The bf16 accumulation that `atbs` needs
  costs nothing visible.
- **int8 per-token drifts but does not break.** The worst case is the chalkboard text
  prompt: the text stays correct and legible, with slightly different letterforms and
  framing. Other prompts show small pose and layout changes.
- The error of these datapaths depends on K (the model's hidden sizes), not on image
  size, so the 512² result carries to 1024².

CPU timing on this box (Ryzen AI 9 HX 370), bf16, 512²: ~11 s per step, ~46-52 s per
image. That is ~1.2 TFLOPS, the bar the NPU pipeline has to clear.

## Attention arithmetic (2026-09-26, 512², 4 steps, 8 prompts)

`attn-fa*` variants route diffusers' Flux2 attention through `fa_emul.dit_fa_attention`,
a model of `open_kernels/designs/dit_fa` that matches the hardware to ~7e-3 (see
`fa_emul.py --check`). The run used the kernel's first version (eager rescale).

| variant (512², 4 steps, 8 prompts) | LPIPS vs bf16 mean / max | PSNR |
|---|---|---:|
| fp32 (noise floor) | 0.0129 / 0.031 | 33.4 dB |
| dit_gemm arithmetic | 0.0150 / 0.031 | 31.7 dB |
| attention, exact exp (`attn-fa-exact`) | 0.0164 / 0.033 | 30.7 dB |
| attention, hardware exp (`attn-fa`) | 0.0220 / 0.067 | 30.2 dB |
| GEMM + attention, hardware exp (the whole NPU DiT) | 0.0288 / 0.081 | 28.3 dB |
| (w8a8 GEMM, for scale: drift, not broken) | 0.0512 / 0.132 | 25.7 dB |

- The aie2p hardware exp2 (a linear-mantissa approximation, +3.8% mean) is the visible
  part: exact exp is at the noise floor, hardware exp drifts small details.
- The whole NPU DiT arithmetic stays well under w8a8's drift; text stays legible.
- dit_fa ships without `FA_EXP_FIX` (+9% attention time) on this result.

~100-150 s per image per attention variant on the CPU (the emulation dominates); the
combined variant ~470 s.

## The VAE decoder (2026-09-26)

- `vae_study.py`: the NPU's VAE arithmetic, emulated on the CPU. It also generates and
  caches the 4-step latents per prompt (`<out>\klein_<size>_s4\latents\`).
- `vae_attn_rank.py`: can the one d = 512 attention head run as dit_fa's head dim 128?
- `capture_vae_goldens.py`: the VAE chain test's inputs (packed bf16 latents) and
  references (fp32 decode, per-stage taps).
- `score_images.py`: LPIPS/PSNR of the chain test's PNGs.

LPIPS against the fp32-math decode of the same latents (512², 8 prompts):

| decode | LPIPS mean / max | PSNR |
|---|---|---:|
| diffusers' bf16 VAE (what the pipeline runs; the checkpoint is bf16) | 0.0000 / 0.0000 | 63.9 dB |
| bf16 activations, fp32 math | 0.0000 / 0.0000 | 65.1 dB |
| NPU arithmetic: bfp16 conv operands + dit_fa attention | 0.0002 / 0.0002 | 54.5 dB |
| attention scores at rank 128, plain SVD (fp32 otherwise) | 0.0004 / 0.0007 | 51.3 dB |
| attention scores at rank 128, data-weighted SVD | 0.0000 / 0.0000 | 72.6 dB |
| **the NPU itself** (`utilities/dit-chain/chain_test_vae.py`) | **0.0014 / 0.0017** | 45.1 dB |
| the NPU itself at 1024² (2 prompts) | 0.0029 / 0.0032 | 44.3 dB |

The DiT's noise floor is 0.013, so the VAE's choices are judged on speed.
