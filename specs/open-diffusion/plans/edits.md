# Phase 8: image edits with FLUX.2 [klein] 4B, every op on the NPU

2026-09-30. Branch `feat/edits` (from `feat/phase7-speed`, PR #141). Status: **in progress.**
Review: `edits-review.md`. The owner took recommendation #1 (serve's flag is now `--imagegen
1`, done in #137) and said "continue with the plan" on 2026-10-01, so the decisions below and
review items #2-#20 are taken as recommended unless noted.

## What klein's edit is (diffusers 0.40.0, `pipeline_flux2_klein.py`, "K" below)

It uses the same `Flux2KleinPipeline` as text-to-image. The `image` argument adds
**reference tokens**:

- **Preprocessing (host):**
  - Resize to ≤ 1024² area, keeping the aspect ratio (LANCZOS). Small images are never
    upscaled.
  - Floor each side to a multiple of 16, then centre-crop, then map to [−1, 1].
  - The output size defaults to the reference's processed size (K:770-782).
- **Encoding:**
  - The VAE encoder produces the latent. The mean is used, not a sample (K:463-476).
  - Patchify 2×2 into 128 channels, then normalise with BatchNorm running statistics
    (`(x − mean) / sqrt(var + 1e-4)`).
  - The result is h·w tokens × 128.
- **The joint sequence:** [text 512 | generated T | reference T_ref], bidirectional
  attention.
  - Position ids: text (0,0,0,l); generated (0,h,w,0); reference i (10 + 10i, h, w, 0).
  - References get the same timestep modulation as the generated tokens. They are never
    updated, but they run through every block at every step (K:844). Their states depend
    on t, so they can't be cached.
- **Output:** only the generated rows go to the scheduler (K:860).
  - `mu`, and so the sigma schedule, counts generated tokens only. It is identical to
    text-to-image at the output size.
- **Not in this pipeline:**
  - no mask and no strength: edits start from pure noise;
  - no guidance: klein is distilled.
  - Inpainting is a separate pipeline (`Flux2KleinInpaintPipeline`: mask, strength,
    blend every step).
  - The KV-cached edit mode belongs to a separately trained 9B checkpoint. It isn't for
    the 4B.

## What's missing on our side

1. **The VAE encoder**, 2.44 TMAC at 1024², 0.56 TMAC at 512².
   - dit_conv has 3×3/1×1 taps at **stride 1 only**; the encoder has three stride-2
     downsamples with (0,1,0,1) padding.
   - Also needed:
     - an RGB8 → bf16 [−1, 1] input op;
     - conv_in with Cin 3 padded to 64;
     - conv_out 512 → 64, of which only the 32 mean channels are needed, with quant_conv
       folded in;
     - a patchify + BN-normalise gather into [T_ref, 128];
     - the mid-block attention (1 head × 512, like the decoder's: reuse the rank-128
       factoring on dit_fa).
   - GroupNorm, SiLU and add (vae_ew) already cover it.
2. **A longer joint sequence.** Every buffer, GEMM M, attention L and ew view is sized by
   T = 512 + T_img today (`klein_pipeline.py:357`).
   - An edit needs new streams at L = 512 + T + T_ref, and the dit_ew `qk` op needs a third
     RoPE branch for reference rows (t = 10, their own grid).
   - Euler and proj_out already touch only the generated rows. If reference rows go last,
     they stay unchanged.
3. **A kernel configuration per edit shape.** ELFs are keyed by resolution
   (`compose_elf.py`, the manifest, `engine.cpp:329`). An edit ELF is another ~20-30 MB,
   and the first edit pays ~2-4 s of kernel creation.
4. **Host input.** OPEN-DIFFUSION-NPU-ONLY doesn't allow decoding or resizing an input
   image. The server throws the uploaded bytes away (`ImageUpload` keeps only metadata).
   There is no image decoder in the tree (only `stb_image_write.h`).
5. **Packaging.** `layout_hash` covers every stream, so any new stream changes it, and
   the published model (`Cyronius/FLUX.2-klein-4B-NPU2`) has to be rebuilt and
   republished along with the encoder weights (~68 MB bf16). Existing installs would be
   refused until re-pulled.

## Estimated speed (from the Phase 7 profile, scaled; not measured)

| configuration (output + reference) | L | step | image |
|---|---:|---:|---:|
| 512² + 512² | 2560 | ~1.6 s | **~7.5 s** (text 0.3 + encoder ~0.2 + 4 steps + decode 0.3) |
| 1024² + 512² | 5632 | ~3.8 s | **~17 s** |
| 1024² + 1024² | 8704 | ~7 s | **~31 s** |

- Attention scales with L², the GEMMs with L.
- At 1024² + 1024², attention is ~3.6× and the GEMMs ~1.9× today's, and activations grow
  from 4.6 GiB to ~9 GiB. This machine has 87.6 GB, so memory isn't the limit.
- For comparison: text-to-image is 3.7 s at 512² and 12 s at 1024².

## Decisions (taken 2026-10-01)

1. **Configurations: 512e512 and 1024e1024** (same size in and out), built as a vertical slice:
   512e512 goes through 8.5 (engine, CLI, server, determinism) before 1024e1024 starts, and the
   model is published once, at the end. 1024 output from a 512 reference is not in this phase.
2. **References are centre-cropped to square, then resized to the configuration's size.**
   - Why square: the DiT side could take other aspects. `dit_fa` already has a `valid_len`
     RTP, so a reference tail-padded to a 512 multiple with its padded keys masked fits one
     DiT configuration for any aspect up to T_ref = 4096 (diffusers caps the area at 1024²).
     The real per-shape costs are the encoder, whose GroupNorm statistics cover the whole
     tensor and so can't run on a padded canvas, and the `qk` op's reference `grid_w`.
   - Growth path, so the API doesn't foreclose it: aspect buckets, each with its own encoder
     streams, plus `valid_len` on the DiT.
   - Consequences, put in the spec and printed by the CLI ("reference centre-cropped W×H →
     S², resized to R"): non-square inputs (most phone photos and screenshots) lose their
     sides and edits come out square; inputs smaller than R are upscaled, where diffusers
     never upscales.
   - Inputs are refused as diffusers refuses them: a side under 64 px, or an aspect over 8:1.
3. **The host decodes and prepares the reference, once per request, before the first NPU
   dispatch, under a pixel cap.** NPU-ONLY is amended to allow it.
   - PNG and JPEG only (`stb_image.h` with `STBI_ONLY_PNG` + `STBI_ONLY_JPEG`). Anything else
     is a 400 naming the format as not implemented.
   - `stbi_info` first. Refuse above 64 MP, and below 64 px a side. The server caps an edit's
     image part at 32 MB, well under its 256 MB body limit.
   - JPEG EXIF orientation (tag 0x0112) is applied, as `diffusers.utils.load_image` does.
   - Centre crop, then `stb_image_resize2` with a custom Lanczos-3 filter (it has no built-in
     Lanczos).
   - **Changed from the first draft:** the host also writes the [−1, 1] bf16 values
     (`bf16(2 · (x / 255) − 1)` in float32, diffusers' arithmetic) into the encoder's input buffer. It is the
     same class of work as writing the seeded noise, and a `vae_ew` `rgb_in` op for it would
     gain nothing measurable.
   - Measure a 12 MP phone JPEG's decode + resize and put it in NPU-ONLY's host-CPU figure.
4. **No masks.** A `mask` part is a 400 naming inpainting as not implemented.
5. **One reference.** A second `image[]` part is a 400 naming multiple references.
6. **Publish the edit model to a new Hugging Face revision, not over `main`.**
   - `oflm pull` downloads only missing files, and a hash mismatch is advisory
     (`model_downloader.cpp:421`, `:618`). Overwriting `main` would leave every existing
     install refused, and would break builds of #136-#141, which read the file list live
     from `tree/main`.
   - `model_list.json`'s `url` / `file_url` point at the new revision (the downloader honours
     `resolve/<rev>`).
   - Upgrade path: document "remove the model directory, then pull". Nothing is on `main`
     yet, so no released install needs more.
7. **The output size follows the reference by default**: the largest configuration not above
   the reference's short side, minimum 512. `--size` overrides it, and the server's
   `size: auto` (OpenAI's edit default) maps the same way.
8. **Base branch: stay on #141.** Its only behaviour change is `FA_TAU` 8 → 32, which the
   long-L spike (8.1.5) validates at the edit lengths anyway.

## Plan

### 8.0 Goldens (1 day)
- `utilities/dit-ref/capture_edit_goldens.py --size 512` (diffusers bf16 on the CPU):
  - References: the 8 study prompts' bf16 images, plus 4 natural photos that are public
    domain or CC0: a 4:3 JPEG carrying an EXIF rotation, a portrait, and one with text.
  - 8 edit prompts ("the same scene at night", "replace the text with …", …), the study's
    noise (seed + i).
  - Saves, per edit: the **preprocessed RGB8 reference** (after diffusers' crop and resize),
    the reference latents (encoder mean, and after patchify + BN), the bf16 edit image, and
    the joint sequence's position ids. Encoder intermediates for edit 0.
  - 2 edits at 1024² (`--size 1024 --prompts 2`).
  - Output: `C:\dev\ditref-out\klein_edit_<R>_s4`.
- The quality gates take the preprocessed RGB8 directly (`generate.py --study`, the chain
  tests), so EDIT and ENCODER measure NPU drift, not resize drift. REFERENCE's tolerance
  covers the resize separately.
- **The CPU emulation of the edit** (`klein_quant_study.py`'s whole-NPU-DiT variant, with
  diffusers' reference latents) predicts EDIT's LPIPS. It emulates the DiT's linears and
  attention, as for generations; the encoder's drift is ENCODER's gate, measured separately.

### 8.1 Spikes (2 days; each can change the design)
1. **Stride-2 conv as space-to-depth.**
   - The downsample reads its input as s2d: a (H/2)×(W/2) grid of 4C channels,
     `D[i, j, (2p + q)C + c] = x[2i + p, 2j + q, c]`. Then `dit_conv`'s 3×3 runs on it with
     zero weights on the −1 taps; the (0,1,0,1) pad lands on the zero border.
   - The residual `add` that feeds each downsample writes D as 4 phase dispatches. Phase
     (p, q) reads `pitch` unchanged, `px_stride` 2C, border 0, offset at (p, q), and writes
     `px_stride` 4C at channel offset (2p + q)C. That add carries no GroupNorm stats, so the
     split costs only 9 extra dispatches per image.
   - 4× the real FLOPs on the three downsamples: ~0.23 TFLOP at 512², ~20 ms.
   - Fallbacks if a view can't express it: a `vae_ew` `s2d` op, then a stride-2 read in
     `dit_conv`'s memtile (~3 days).
2. **Reference RoPE in `qk`** (a verification task). A reference row is an image row with
   axis 0 rotated by t = 10, which is `FINE[10]` in the existing table. In same-size
   configurations an image token k ≥ grid_w² is reference token k − grid_w², so no RTP is
   added. Check it fits the 16 KB program memory.
3. **The encoder attention at rank 128:** its accuracy on the encoder's weights
   (`vae_attn_rank.py`).
4. **conv_out's output:** a `dit_conv` output with no border, or a latent gather that reads a
   bordered buffer.
5. **The long-L streams.** Before 8.2:
   - `dit_fa` standalone at L = 2560 and 8704, 24 heads, against `fa_emul.py`. 8704 walks
     136 key chunks per pass against 72 validated, and the lazy rescale's τ = 32 is
     validated only to 4608.
   - `dit_gemm` `sgl_in` / `sgl_out` at M = 8704; `dit_ew` views at T = 8704.

### 8.2 The encoder on the NPU (4-5 days)
- **Schedule:** `open_kernels/vae_encoder.py`, beside `vae_decoder.py`:
  - conv_in (Cin 3 padded to 64, reading the host-written [−1, 1] buffer);
  - 4 down blocks (resnets as the decoder's, stride-2 per 8.1.1);
  - mid (resnet, attention, resnet); GroupNorm + SiLU; conv_out ⊗ quant_conv (32 mean
    channels, Cout padded to 128);
  - the latent gather (patchify 2×2 + BN normalise → [T_ref, 128]), per 8.1.4.
- **Weights:** `pack_weights` gains the `encoder.*` keys.
- **Chain test:** `utilities/dit-chain/chain_test_vae_enc.py`, from 8.0's preprocessed RGB8,
  against diffusers' encoder mean.
  - Gate: latent rel_fro, plus the LPIPS of decode(NPU latent) against decode(diffusers
    latent). The decoder's gate was LPIPS 0.0014.

### 8.3 The DiT's edit configuration (4-5 days)
- **Schedule:** `klein_pipeline.plan(R, ref=True)` makes the joint sequence [txt | gen | ref]
  and sizes every buffer and stream by L = 512 + 2T.
  - Reference rows belong to the "img" part: the double blocks' img streams run at
    M = 2T, with the generated tokens' modulation.
  - Each step: x_emb on LAT into the generated rows, x_emb on REFLAT into the reference
    rows. **REFLAT's x_emb runs every step, though its result never changes:** the blocks
    overwrite X's reference rows, so caching it would read stale rows. It costs 1.6 GMAC per
    step at T_ref = 4096.
  - The `qk` op takes reference rows per 8.1.2.
  - Euler and proj_out stay unchanged: they touch only the generated rows.
- **Exporter:** streams for the new L (sgl_in/out, attn_dbl/sgl, ln/res/qk views).
- **Chain tests:** one double block and one single block with reference rows, against the
  emulation.
- **End to end:** `generate.py --ref <rgb8>` runs the whole edit, and its quality gate runs
  against 8.0's goldens.

### 8.4 Kernel configurations and packaging (2 days)
- **Keys:** `compose_elf.py` and the manifest key ELFs by configuration (`"512"`, `"1024"`,
  `"512e512"`, `"1024e1024"`), and so does the engine's lookup.
- **Model:** `export_bundle` ships the encoder weights and the edit schedules;
  `q4nx-build --open-diffusion` builds them.
  - Upload to a new revision of `Cyronius/FLUX.2-klein-4B-NPU2` (decision 6), point
    `model_list.json` at it, update `model_info.json`'s hashes, pull clean.
  - Install the kernels into `src/xclbins` and `src/out/xclbins`.
- **Resident memory:** state the total for an engine that has selected all four
  configurations (activations + 7.5 GB of weights, pinned), and either confirm it fits the
  NPU-visible budget beside a resident chat model or evict the least recently used
  configuration's activations.

### 8.5 Engine, CLI, server (3 days)
- **Engine:** `select_edit(R)`, then `set_reference(bf16)`. The encoder runs as an `encode`
  phase before the steps, in the same hardware context.
- **Host:** `prompt.cpp`'s neighbour `reference.cpp` decodes, applies EXIF orientation,
  centre-crops to square and resizes to R (decision 3). It is pure host code and
  unit-tested.
- **CLI:** `oflm image <tag> "<prompt>" --image in.png [--size 512|1024]`. `--image <file>`
  is refused by every other command, like `-o`, `--size` and `--seed`.
- **Server:**
  - `ImageUpload` keeps the bytes.
  - `/v1/images/edits` runs one `image` / `image[]` part; `size: auto` follows the reference.
  - `mask` or a second image → 400, naming what isn't implemented.
  - HRX builds keep 501.

### 8.6 Specs, tests, docs (1-2 days)
Update the spec (below), write the tests, and create an `open-diffusion-edits` skill. Then
merge the durable parts into `spec.md` and archive this plan.

**Total: ~3 weeks.** The risks are 8.1.1's stride-2 route and 8.1.5's long-L streams. If the
s2d views fail, a real stride-2 read in `dit_conv`'s memtile is ~3 more days.

## Spec impact

**New requirements:**
- **OPEN-DIFFUSION-EDIT** (manual): `oflm image --image` and `/v1/images/edits` produce an
  edit conditioned on one reference, with diffusers' sequence and positions.
  - Gate: LPIPS against diffusers' bf16 edit, near the emulation's prediction; the images
    coherent.
  - **Ablation:** LPIPS(NPU edit, diffusers edit) is well below LPIPS(NPU edit with a
    different reference, same seed, diffusers edit). A DiT that ignores the reference still
    makes coherent images; this catches it.
- **OPEN-DIFFUSION-ENCODER** (manual): the NPU encoder's latents against diffusers' mean, on
  the study references and the natural photos. Gate: rel_fro and the decoded LPIPS.
- **OPEN-DIFFUSION-REFERENCE** (test): the host's reference preparation. Acceptance criteria:
  - a 512² PNG at R = 512 passes through byte-identical;
  - a 4000×3000 JPEG gives the centred 3000² crop box, resized to R;
  - the resize lands within ±N levels per channel (max) of PIL's crop + LANCZOS on the same
    crop (N set from the first measurement, then fixed);
  - RGBA drops alpha (PIL's `convert("RGB")`); grayscale and palette images become RGB;
  - EXIF orientation 6 rotates;
  - too small (< 64 px a side), too large (> 64 MP), over 8:1, and undecodable or
    non-PNG/JPEG inputs are refused, naming the reason.

**Modified requirements:**
- **OPEN-DIFFUSION-NPU-ONLY:** the host may also decode, orient, crop and resize the
  reference and write its [−1, 1] bf16 values, once per request, before the first NPU
  dispatch, under the pixel cap.
- **OPEN-DIFFUSION-CLI:** `--image <file>`; the size follows the reference by default.
- **OPEN-DIFFUSION-DETERMINISM:** an edit twice gives the same bytes; engine = pyxrt for one
  edit at 512².
- **OPEN-DIFFUSION-PACKAGE:** encoder weights, the edit configurations, the
  configuration-keyed manifest, the pinned revision.
- **OPEN-DIFFUSION-RESOLUTIONS:** the edit configurations. Criteria: (512, 512) and
  (1024, 1024) pass; any other (R, R_ref) pair is refused, naming the supported pairs.
- **OPEN-DIFFUSION-STEPS:** edits accept 1-50 steps like generations. REFLAT's x_emb
  arguments have stride 0 across steps, which the engine's step-stride derivation must
  accept.
- **OPEN-DIFFUSION-PERF:** the edit times, with "first edit after load" as its own row.
- **SERVER-IMAGES-EDITS** (`specs/server-api/spec.md`): implemented for one image. The
  criterion at `:368` changes: two `image[]` parts are a 400 naming multiple references; a
  `mask` is a 400 naming inpainting; one image returns a PNG; HRX still 501.

**Removed:** none.

## Progress

### 2026-10-01
- **8.0 goldens:** `utilities/dit-ref/capture_edit_goldens.py`. 12 bf16 edits at 512²
  (`C:\dev\ditref-out\klein_edit_512_s4`): the 8 study images, plus scikit-image's astronaut
  (public domain), coffee (CC0, written as an EXIF-6 JPEG), chelsea (CC0, 451×300) and
  text (public domain, grayscale). The `npu-emul` variant and the 2 edits at 1024² are
  queued.
- **8.1.1 stride 2: passes** (`utilities/dit-chain/spike_s2d.py`). The 4 phase adds write D
  bit-exactly with its border zero, and the s2d conv matches diffusers' stride-2 formula at
  rel_fro 0.8-1.2e-2 on all three 512² downsample shapes. No new op.
- **8.1.2 reference RoPE: passes.** It is 6 lines in `ew.cc`, with no RTP: an image token
  k ≥ grid_w² is reference token k − grid_w², and axis 0 turns by `FINE[10]`.
  `make_test.py --op qk --edit` gives rel_fro 2.6e-3 on the reference rows.
- **8.1.3 encoder attention at rank 128: kept.** In fp32 it costs decoded LPIPS 0.0002 and
  8.6% latent rel_fro (`vae_attn_rank.py --encoder`). On the NPU, the edits barely move
  with diffusers' own tokens in its place: LPIPS 0.0140 against 0.0133 (below).
- **8.1.4 conv_out:** one 3×3 conv on norm_out's space-to-depth grid computes the packed,
  BN-normalised tokens (`latent_out_weights`). A vae_ew add of a zero buffer copies its
  bordered output into the plain REFLAT. vae_ew gained an `npix` spec key for the 4 phase
  gn_applys.
- **8.1.5 long L: passes.** `dit_fa` at L = 2560 and 8704 (24 heads): rel_fro 2.5e-2 and
  2.75e-2. `dit_gemm` sgl_in / sgl_out / img_qkv at M = 8704 / 8192 are bit-identical to the
  validated 1024 streams (`utilities/dit-chain/spike_long_m.py`).
- **8.2 encoder: done** (`open_kernels/vae_encoder.py`, `chain_test_vae_enc.py`). 89
  dispatches. Tokens rel_fro 11.7% against the fp32 mean (diffusers bf16: 1.0%); decoded
  LPIPS 0.0008, max 0.0010 (`score_ref_latents.py`). `encoder_study.py` splits the error:
  the rank-128 attention, and dit_conv's bf16 accumulator re-rounding, which the emulation
  leaves out (the decoder shows the same 7× gap).
- **8.3 DiT edit configuration: done** (`klein_pipeline.plan(R, edit=True)`, the exporter's
  `--edits`). The text-to-image plans and layout hash are unchanged (`e0450140c44f78e0`).
  The edit plan has 1143 dispatches and 1.70 GiB of activations at 512e512, 5.58 GiB at
  1024e1024: encoder buffers shaped like decoder ones alias them.
- **First NPU edits** (`generate.py --edit`, `C:\dev\klein-kernels-edit`), 12 at 512² against
  diffusers bf16:
  - NPU everything: LPIPS 0.0133 mean, 0.041 max, PSNR 33.1 dB;
  - diffusers' reference tokens in place of the NPU encoder's: 0.0140;
  - **ablation**, each edit given the next edit's reference: 0.7317.
- **8.5 host:** `src/open_diffusion/reference.cpp`. stb_image (PNG and JPEG only), the EXIF
  orientation, and a port of PIL's resample in place of stb_image_resize2: stb's single
  float pass differed from PIL by up to 28 levels at hard edges and image borders.
  `specs/open-diffusion/tests/test_reference.py`: 28 tests. A PNG resize is within 1
  level of PIL, all 8 EXIF orientations match exactly, and the refusals are named.
- **8.5 engine:** configuration keys (`select(size, steps, edit)`, `set_reference`,
  `edit_sizes`), and `open_diffusion_cli --ref`. Built; not yet run.
- **8.5 done for 512e512.**
  - `oflm image <tag> "<prompt>" --image FILE`. The size follows the reference unless
    `--size` is given; the reference is prepared before anything loads.
  - `/v1/images/edits`: one image is edited; two images, a mask or a non-PNG/JPEG file are a
    named 400.
  - Engine == pyxrt bit for bit, for an edit and for text-to-image.
  - Tests: `test_images_api.py`, 10 edit and generation tests pass; `test_cli_determinism`
    and `test_engine_matches_pyxrt` each gained an edit case. All pass against the scratch
    install: `C:\dev\oflm-models-edit` (a junction to `C:\dev\klein-bundle-edit`) and
    `C:\dev\klein-kernels-edit-inst`.
- **Found: the SVD sign wasn't reproducible across numpy builds.**
  `vae_decoder.attention_weights` (both coders use it) gave one singular pair the opposite
  sign under numpy 2.5.3 than under 2.4.6. bfp16's int8 mantissa isn't sign-symmetric, so
  pixels moved by up to 10 levels: a bundle packed in one environment didn't match the
  pyxrt cache packed in the other. Fixed by canonical signs (each k' column's largest entry
  positive); now identical across both numpy versions.
  - The published model's `vae_W.bin` will change at republish. Its pixels move by the same
    ≤ 10 levels; LPIPS is unaffected.
- **Found and fixed (later the same day): the one-context engine failed above 8 steps.**
  The cause was the configure runs, one per stretch (~140 per step), each with a whole
  set's control code. They now go back to a free list when their stretch completes; with
  that, two-set rebinding works too (the page fault below was the configure runs running
  out). Plus the DT sizing at exactly 50 steps. 1-50 steps run, for text-to-image and edits.
  The original notes:
- **Found, pre-existing in #140: the one-context engine fails above 8 steps.**
  - Text-to-image at 9+ steps and edits at 8+ fail with the driver's "Cannot extend beyond
    8 banks" (`xrt_core.dll`), at HEAD too. OPEN-DIFFUSION-STEPS promises 1-50.
  - Reusing two sets of step runs (rebinding per step) moved the limit by one step, then
    page-faulted the device at 10. That change was reverted: the binding resource isn't the
    op runs. The cfg runs (~140 per step) or the per-step sub-buffers are the next suspects.
  - Edits run at 1-7 steps.
- **Provisional timing** (another session's CPU job at ~50-70% load, no turbo): 512²
  text-to-image 6.6-6.9 s, edit 10.4-11.5 s (encode 0.35 s, steps 1.7× text-to-image's).
  Re-time on a quiet machine.
- **1024e1024: done.** `--edits 512,1024` added 64 streams and `diffusion_r1024e1024.elf`
  (29.4 MiB); the set installs 4 ELFs (85.9 MiB), layout `e667e5952ca22004`.
  - 2 edits vs diffusers bf16: LPIPS 0.0072 mean / 0.0088 max; ablation 0.84.
  - Encoder: decoded LPIPS 0.0013, tokens 11.3%.
  - Engine == pyxrt at 1024.
  - Under 100% CPU load: 62-66 s per edit, ~14 s per step, every kernel ~2× its quiet rate.
    Scaled to quiet: ~7.4 s per step, ~32 s per edit, as estimated.
- `src/model_list.json`: `image_edit_sizes` [512, 1024]. Its `files` and `url` change at the
  publish (decision 6), which waits for the owner's OK.
- **8.4 published (2026-10-01, owner's OK).**
  - Branch `edits` of `Cyronius/FLUX.2-klein-4B-NPU2`, commit `d84e3fd6`: 21 files, layout
    `e667e5952ca22004`; `main` unchanged.
  - `model_list.json` pins `resolve/<sha>`; `model_info.json` holds the hub's listing.
  - The kernel set is installed in `src/xclbins` and `src/out/xclbins`.
  - Clean `oflm pull` verified all 21 files; `oflm image`, with and without `--image`, ran
    from it.
  - PR #146 (stacked on #141).
