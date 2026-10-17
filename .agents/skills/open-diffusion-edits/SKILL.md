---
name: open-diffusion-edits
description: Build, run, verify and extend FLUX.2 [klein] 4B image edits on the XDNA2 NPU -- the VAE encoder schedule (open_kernels/vae_encoder.py: stride-2 downsamples as space-to-depth over existing dit_conv/vae_ew streams), the [text | generated | reference] DiT configuration (klein_pipeline.plan(R, edit=True), streams r<R>e<R>_*), the host reference preparation (src/open_diffusion/reference.cpp, a port of PIL's LANCZOS) and the edit paths of oflm image --image and /v1/images/edits. Use when changing edits, adding an edit size or a non-square/aspect-bucket reference, debugging an edit that ignores its reference or drifts, or regenerating the edit goldens.
---

# FLUX.2 [klein] edits on the NPU

Plan and history: `specs/open-diffusion/plans/edits.md` (archived to `archive/` when done).
Spec: OPEN-DIFFUSION-EDIT, -ENCODER, -REFERENCE (and the edit parts of RESOLUTIONS, CLI,
DETERMINISM, STEPS) in `specs/open-diffusion/spec.md`; SERVER-IMAGES-EDITS in
`specs/server-api/spec.md`. The generation pipeline itself is the `open-diffusion` skill.

## What klein's edit is (diffusers 0.40 `Flux2KleinPipeline(image=...)`)

- The reference's VAE latents become T extra tokens after the generated ones: the
  encoder's **mean** (argmax), patchified 2x2 (channel 4c + 2dy + dx), then BN-normalised
  with `bn.running_mean/var`, eps 1e-4.
- Their positions are (t = 10, h, w, 0).
- They get the generated tokens' timestep modulation, run through every block each step,
  and are sliced off before the scheduler.
- mu, and so the sigmas, count the generated tokens only: the same as text-to-image at R.
- No mask, no strength, no CFG. Inpainting is a different pipeline; the KV-cached edit
  mode is the 9B's.

## Files

- `open_kernels/vae_encoder.py`: packing (`pack_weights`, `s2d_weights`,
  `latent_out_weights`) and the schedule (89 dispatches): conv_in, 4 down blocks, mid
  (attention at rank 128), norm_out, then latent_out.
- `open_kernels/klein_pipeline.py`:
  - `plan(R, edit=True)` and `check_edit` / `config_key` / `parse_config`.
  - `encoder_buffers` aliases encoder buffers onto decoder buffers of the same shape. That
    is how 1024e1024 fits in 5.6 GiB.
- `open_kernels/export_dit_kernels.py --edits 512,1024`; `compose_elf.py` makes
  `diffusion_r<R>e<R>.elf`.
- `open_kernels/designs/dit_ew/ew.cc` `rope_token`: an image token k ≥ grid_w² is reference
  token k − grid_w², axis 0 rotated by `FINE[REF_T = 10]`. No RTP.
- `open_kernels/designs/vae_ew/vae_ew.py` `npix`: gn_apply over a quarter of the pixels its
  stats covered (the space-to-depth phases).
- `utilities/dit-chain/`:
  - `chain_test_vae_enc.py [--build]`: the encoder alone, vs the fp32 encoder;
  - `generate.py --edit --study <klein_edit dir>`, with `--ref-tokens` (diffusers' tokens:
    isolates the DiT) and `--swap-ref K` (the ablation);
  - `spike_s2d.py`, `spike_long_m.py`: the 8.1 spikes, rerunnable.
- `utilities/dit-ref/`:
  - `capture_edit_goldens.py`: references, encoder taps, tokens, bf16 edits, the
    `npu-emul` / `ref-npu` / `enc-wsvd128` variants;
  - `score_ref_latents.py`: the decoded-LPIPS gate;
  - `encoder_study.py`: the NPU arithmetic emulated for the encoder;
  - `vae_attn_rank.py --encoder`.
- `src/open_diffusion/reference.{hpp,cpp}`: decode (stb_image, PNG/JPEG only), EXIF,
  centre crop, PIL's resample, the bf16 map. `reference_tool.cpp` exists for the test.
- Engine: `select(size, steps, true)`, `set_reference(rgb)`, `edit_sizes()`.
  `open_diffusion_cli --ref FILE|.npy`.
- `oflm image ... --image FILE` (`src/src/image_command.hpp`), `/v1/images/edits`
  (`src/server/rest_handler.cpp`), the registry's `image_edit_sizes`.

## Run it

```
. C:\dev\mlir-aie\iron_env.ps1
python open_kernels\export_dit_kernels.py --resolutions 512,1024 --edits 512,1024 --out C:\dev\klein-kernels-edit --jobs 6
python utilities\dit-chain\generate.py --kernels C:\dev\klein-kernels-edit --size 512 --edit --study C:\dev\ditref-out\klein_edit_512_s4 --prompts 12 --out <d>
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_images.py <d> --test "{:02d}.png" --ref "{:02d}.png" --ref-dir C:\dev\ditref-out\klein_edit_512_s4\bf16 --n 12
```

Expected at 512² (2026-10-01):
- LPIPS 0.0133 mean / 0.041 max vs diffusers bf16;
- the `--swap-ref 1` ablation: 0.73;
- `--ref-tokens`: 0.0140.

If the ablation number falls toward the real one, the DiT is ignoring the reference.
Suspect the qk reference branch or the reference rows' x_emb.

## What was learned (don't re-derive it)

1. **Stride 2 needs no new kernel.** The residual add before each downsample runs as 4
   phase dispatches:
   - read view: `pitch` unchanged, `px_stride` 2C, border 0, offset at (p, q);
   - write view: D's `px_stride` 4C, at channel offset (2p + q)C.

   That builds the space-to-depth grid. dit_conv's 3x3 then runs on it with zero −1 taps;
   the right/bottom pad lands on D's zero border. It costs 4× FLOPs on three convs (~20 ms
   at 512²). The same trick writes norm_out's output for latent_out.
2. **conv_out → tokens is one conv.** conv_out, quant_conv's mean half, the patchify and the
   BN fold into one 3x3 conv on norm_out's space-to-depth grid (`latent_out_weights`;
   output channel 4c + 2dy + dx reads s2d row i + a, phase p with 2a + p = dy − 1 + ky).
   dit_conv only writes bordered outputs, so a vae_ew add of a zero buffer copies them
   into the plain REFLAT that x_emb reads.
3. **The encoder's token error is large and harmless.**
   - Tokens are 11.7% rel_fro from the fp32 mean (diffusers bf16: 1%). The causes are the
     rank-128 attention and dit_conv's bf16 accumulator re-rounding every 64 of K, which
     no emulation models; the decoder shows the same 7× gap.
   - Decoded LPIPS is 0.0008, and the edits don't move with diffusers' tokens in place
     (0.0140 vs 0.0133).
   - Judge the encoder by decode and by edit LPIPS, never token rel_fro alone. The
     data-calibrated factorization (3% rel_fro) was not needed.
4. **Reference rows are rewritten every step.** x_emb on REFLAT runs each step because the
   blocks overwrite X's reference rows. Caching it reads stale rows.
5. **Long sequences just work.** dit_fa at L = 8704 (136 key chunks per pass): rel_fro
   2.75e-2. dit_gemm at M = 8704 is bit-identical to the 1024 streams on overlapping rows.
   A stream's rows are independent, so `spike_long_m.py`'s differential test needs no
   reference arithmetic.
6. **Match PIL, not "Lanczos".** stb_image_resize2 differs from PIL by up to 28 levels at
   hard edges and borders: PIL clips to uint8 between its two passes and truncates and
   renormalizes the kernel at edges. `reference.cpp` ports PIL's resample and is within 1
   level everywhere.
7. **SVD signs are LAPACK's choice.** numpy 2.4 and 2.5 flipped one singular pair of the
   attention factorization. bfp16's int8 mantissa is asymmetric, so a flip moved pixels up
   to 10 levels, and a bundle packed in one environment stopped matching a pyxrt cache
   packed in another. `attention_weights` fixes each pair's sign. When engine ≠ pyxrt by a
   few levels, diff `vae_W.bin` against `vae_packed.npz` first.
8. **PNG bytes from the engine (stb_image_write) and generate.py (zlib) differ for equal
   pixels.** Compare decoded pixels.
9. **The driver gives every XRT run its own copy of its control code** from a bounded heap
   ("Cannot extend beyond 8 banks"; past it, a device page fault). Configure runs carry a
   whole set's register writes. The engine recycles them through a free list, and keeps
   two sets of step runs that it rebinds per step. Before that fix, 9+ steps failed. Don't
   add runs per stretch or per step.
10. **Timing needs a quiet machine.** Another session's CPU job doubled the numbers. Check
    `Get-Process` and the load first.

## Adding an edit size, or non-square references

- A same-size edit at a new R needs only `--edits R`. The encoder and DiT streams follow
  from the plans; check `check_edit`, the encoder's vae_ew element counts, and dit_conv's
  tile count (a multiple of 8).
- An output size different from the reference's needs RTPs in `qk` (`n_gen` and the
  reference's `grid_w`).
- Aspect buckets: encoder streams per bucket (GroupNorm statistics cover the whole tensor,
  so no padded canvas), plus `valid_len` on the DiT's attention to mask the padded
  reference keys.
