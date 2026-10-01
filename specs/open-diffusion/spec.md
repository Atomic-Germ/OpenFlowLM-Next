# Open diffusion: FLUX.2 [klein] 4B text-to-image and edits on the NPU

What the open image engine must do, and what has been measured. Directory name gives the
prefix: `OPEN-DIFFUSION`.

The model is `black-forest-labs/FLUX.2-klein-4B` (distilled, 4 steps, no CFG):
- the DiT has 5 double-stream and 20 single-stream blocks (hidden 3072, 24 heads of 128);
- the text encoder is Qwen3-4B layers 1-27 (taps 9/18/27);
- the VAE decoder is AutoencoderKLFlux2;
- an edit adds its VAE encoder: klein's own pipeline with one reference image, whose latent
  tokens follow the generated ones in the joint sequence (`specs/open-diffusion/plans/edits.md`).

**The reference is the diffusers pipeline in bf16 on the CPU**
(`utilities/dit-ref/klein_quant_study.py`, 8 fixed prompts, seed 1234 + i). The goal set
by the user is maximum speed without broken images: drift is acceptable, breakage is
not.

The schedule is one list, `open_kernels/klein_pipeline.py`:
- text encoder, conditioning, 4 steps, VAE;
- 1050 dispatches over six kernel sets;
- an edit (`plan(R, edit=True)`, configuration `<R>e<R>`) adds the VAE encoder
  (`open_kernels/vae_encoder.py`) before the steps: 1143 dispatches at 512².

The exporter builds its streams (`open_kernels/export_dit_kernels.py`) into six kernel sets,
then assembles those into one full ELF per resolution (`open_kernels/compose_elf.py`). Two
runners replay the same list:
- `utilities/dit-chain/generate.py` (pyxrt), on the six sets' own xclbin contexts;
- `src/open_diffusion/` (native), in one hardware context per resolution, from the model
  directory `utilities/dit-chain/export_bundle.py` writes (`q4nx-build --open-diffusion`).
  `oflm image` and `oflm serve` run it.

## Requirements

### OPEN-DIFFUSION-NPU-ONLY: no host compute that grows with the data
**Applies to:** `open_kernels/klein_pipeline.py`, `utilities/dit-chain/generate.py`, `src/open_diffusion`, `oflm image`
**Verification:** manual

During a generation the host may do only these things:
- tokenize the prompt;
- gather the prompt's 512 embedding rows;
- write the seeded noise;
- set `te_attn`'s `valid_len`: the pyxrt runner patches its instruction words, the native
  engine picks the prompt length's head kernel (`fa:te_attn_vl<n>`);
- for an edit, prepare the reference once per request, before the first NPU dispatch:
  decode, orient, crop and resize it (OPEN-DIFFUSION-REFERENCE; refused above 64 MP), and
  write its bf16 2 (x / 255) - 1 values;
- read the RGBA and encode the PNG or JPEG.

Everything else is an NPU dispatch. In the native engine every run of a resolution goes to
one hardware context, queued. The host blocks, without polling, only at phase boundaries,
when its window of runs in flight is full, and on the image's last run. The pyxrt runner
queues runs within one kernel set and blocks on the last one before switching sets.

**Verification (manual):** run `generate.py` (or the native CLI) on a quiet machine and
compare the process CPU time per image with the NPU wall time. The report prints both
(`host CPU`).

**Measured 2026-09-27** (pyxrt runner): 0.1-0.45 s of host CPU per image, against 6.5-7.4 s
(512²) and 20 s (1024²) of NPU time.

**Measured 2026-09-29** (native engine, one context): 0.1-0.2 s of host CPU per image at
512² and 0.14-0.34 s at 1024², against 3.7 s and 12.0 s of NPU time.

### OPEN-DIFFUSION-QUALITY: not broken, and within the predicted drift
**Applies to:** the whole pipeline
**Verification:** manual

The runner is given the study's fixed noise and prompts (`capture_pipeline_inputs.py`,
`generate.py --study`). Its images must meet all of these:
- they are coherent;
- text prompts render legible text;
- the LPIPS against the bf16 CPU run lands near what the CPU emulation of the NPU
  arithmetic predicts. Materially higher is a bug.

**Verification (manual):**

```
python utilities\dit-chain\generate.py --kernels <set> --size 512 --study C:\dev\ditref-out\goldens_pipe_512 --out <dir>
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_images.py <dir> --test "{:02d}.png" --ref "{:02d}.png" --ref-dir C:\dev\ditref-out\klein_512_s4\bf16
```

Then look at the grid.

**Measured 2026-09-27**, 512², 8 prompts, LPIPS vs bf16 (noise floor: fp32 vs bf16 is
0.013):

| run | LPIPS mean / max | CPU-emulated prediction |
|---|---|---|
| DiT + VAE on the NPU, the bf16 text embeddings | 0.044 / 0.092 | 0.029 / 0.081 (DiT linears + attention only) |
| the same with diffusers' exact modulation vectors injected | 0.047 / 0.180 | |
| **everything on the NPU** | **0.107 / 0.191** | te-npu alone: 0.092 / 0.241 |

- All 16 images are coherent. "OPEN LATE" and "SOUP OF THE DAY: TOMATO" render legibly.
- At 1024² (2 prompts) the images are coherent too: LPIPS 0.33 and 0.15. Prompt 0's sign
  moves within the frame.
- The gap to the DiT prediction is the conditioning GEMMs. The study did not emulate
  them: timestep MLP, modulation and embedders. They share dit_gemm's bf16-accumulator
  arithmetic, which puts the modulation vectors 2-3% from diffusers'.
  - Injecting diffusers' exact vectors takes the 7 non-chaotic prompts from 0.038 to
    0.028.
  - Prompt 0 swings either way under any perturbation.
- The rest of the full pipeline's drift is the text encoder's padding rows. That drift
  is known, accepted, and not NPU-specific (`utilities/dit-chain/README.md`).

### OPEN-DIFFUSION-RESOLUTIONS: the supported sizes, and a named refusal for others
**Applies to:** `open_kernels/klein_pipeline.py`
**Verification:** test
**External tests:** none (`specs/open-diffusion/tests/test_host_setup.py`)

A square size R runs only when all of these hold:
- R is a positive multiple of 16 px;
- (R/16)² image tokens is a multiple of 512 (dit_gemm's M tile).

Any other size is refused with the reason named, never run wrong.

An edit is an R x R output from one R x R reference (`check_edit(R, R_ref)`): dit_ew's qk op
takes the reference tokens as the image tokens past grid_w², on the same grid. Its joint
sequence is [text | generated | reference], 512 + 2 (R/16)² rows.

**Acceptance criteria:**
- 512 and 1024 pass `check_resolution`.
- 768 is refused because 2304 image tokens is not a multiple of 512.
- 520 and 0 are refused as not a positive multiple of 16.
- `plan(768)` raises.
- (512, 512) and (1024, 1024) pass `check_edit`. (1024, 512) and (512, 1024) are refused as
  needing the reference at the output's size; (768, 768) for its token count.
- `plan(512, edit=True)`: X holds 512 + 2T rows. The encoder phase precedes step 0. Each step
  writes the reference rows (from row 512 + T) out of REFLAT. proj_out reads the T generated
  rows only.
- A text-to-image plan has no REFLAT and no encode phase.

### OPEN-DIFFUSION-SCHEDULE: the scheduler matches diffusers
**Applies to:** `open_kernels/klein_pipeline.py`
**Verification:** test

The host computes the flow-match sigmas: the exponential time shift with FLUX.2's
empirical mu, and a terminal 0. They must match diffusers' FlowMatchEulerDiscreteScheduler
bit for bit, because a changed sigma changes every image silently. Euler's dt reaches the
NPU as an fp32 in its parameter run.

**Acceptance criteria:**
- `sigmas(512)` = [1.0, 0.95808536, 0.88398188, 0.71749657, 0.0] (float32, exact).
- `sigmas(1024)` = [1.0, 0.96738404, 0.90814394, 0.76719993, 0.0].
- `dt_params` stores sigma[s+1] - sigma[s] as the fp32 at the parameter run's first
  vector.

### OPEN-DIFFUSION-STEPS: any step count from 1 to 50, the default unchanged
**Applies to:** `src/open_diffusion/engine.cpp`, `src/open_diffusion/schedule.hpp`
**Verification:** test

klein is distilled for 4 steps and the bundle is exported with 4, but the streams allow any count
up to 512 (the modulation GEMM's M). `Engine::select(size, steps)` runs `steps` denoising steps:
step k is the bundle's step 0 with each argument moved on by its stride from step 0 to step 1
(the engine derives the strides and refuses a schedule whose steps do not lie on one line). Only
the modulation views (368,640 bytes per step) and the Euler dt views (24,576) move. A count other
than the bundle's gets its timestep features and dts from `schedule.hpp`, klein_pipeline's
`sigmas` / `timestep_features` / `dt_params` in C++. For the bundle's own count the engine writes
the bundle's `tf_<R>.bin` / `dt_<R>.bin`, so the default image does not change.

One `Engine` serves every resolution the bundle has: the weights and kernel sets load once, and
a resolution's activations are allocated the first time it is selected.

**Acceptance criteria** (the `open_diffusion_schedule` CTest, `src/open_diffusion/schedule_test.cpp`,
against the installed model; it fails naming the path without it):
- At 512 and 1024, 4 steps: `dt_params` equals `dt_<R>.bin` word for word, and
  `timestep_features` is within 2^-8 of `tf_<R>.bin` in every word. (5 of 131,584 words differ
  by one rounding: numpy's float32 `exp` is not correctly rounded, and schedule.hpp uses double.)
- `sigmas(4096, 8)` has 9 values from 1 down to 0, strictly decreasing.

**Verification (manual)**, once, 2026-09-28: after the change, `oflm image flux2-klein:4b "a red
fox in fresh snow" --size 512 --seed 1` and `"a lighthouse at dusk" --size 1024 --seed 7 -o
x.jpg` wrote files byte-identical to the previous engine's. Through the server at 512², seed 1,
the same fox: `steps: 2` softer (3.6 s), `steps: 8` sharper (10.4 s), both coherent.

Edits take the step count the same way: the reference rows' x_emb arguments do not move
between steps (stride 0), which the step-line derivation accepts.

**Fixed 2026-10-01** (broken since the one-context engine of #140): above 8 steps for
text-to-image (7 for a 512 edit), the driver refused the engine's runs with "Cannot extend
beyond 8 banks". The driver gives each XRT run its own copy of its control code from a
bounded heap. The engine had made one configure run per stretch (~140 per step), each a
whole set's register writes, and one set of op runs per step. Now:
- a configure run returns to a free list once its stretch completes;
- two sets of step runs are rebound per step (their moving arguments only), after the step
  that last used the set has drained.

DT is also sized for kMaxSteps whole slots; 50 steps needed one slot more than the last
view.

**Verification (manual)**, 2026-10-01, `open_diffusion_cli`, 512²:
- text-to-image at 12, 20 and 50 steps, and the edit at 8 and 50, are coherent;
- 12 steps with rebinding equals 12 steps with one set per step, byte for byte;
- 4 steps equals pyxrt, for text-to-image and for the edit.

### OPEN-DIFFUSION-CLI: `oflm image` writes one image
**Applies to:** `oflm` (`src/src/image_command.hpp`, `src/include/utils/vm_args.hpp`, `src/include/AutoModel/model_families.hpp`)
**Verification:** manual

`oflm image <tag> "<prompt>" [-o FILE] [--size 512|1024] [--seed N] [--image FILE]` writes
one image and prints its path, the seed and the time on the NPU (text, steps, VAE; for an
edit, the encoder too). With `--image` it is an edit of that reference (OPEN-DIFFUSION-EDIT):
- the reference is prepared and checked (OPEN-DIFFUSION-REFERENCE) before anything is
  downloaded or loaded, and the CLI prints what was done to it ("reference 4000x3000
  centre-cropped to 3000x3000, resized to 512x512");
- the size defaults to the largest of the registry entry's `image_edit_sizes` not above the
  reference's shorter side, or the smallest of them; `--size` overrides it;
- an entry with no `image_edit_sizes` is refused as having no edit configurations.
- The format comes from the extension: `.png`, `.jpg` or `.jpeg`, any case. Any other is
  refused before loading. The default file is `oflm-<seed>.png` in the current directory.
- `--size` defaults to 1024. A size the registry entry's `image_sizes` does not list is
  refused before any download, naming the supported ones.
- `--seed` defaults to a random 64-bit value.
- A tag whose registry entry lacks `"image": true` is refused before any download.
- `-o`, `--size`, `--seed`, `--image` and a third positional are refused by every other
  command.
- `oflm run` and `oflm serve` refuse the image tag as not a chat model, before unloading
  anything. `/api/tags` does not list it; `/v1/models` does, for the Images API
  (SERVER-IMAGES-GENERATIONS in `specs/server-api/spec.md`).
- An HRX build answers that `oflm image` is not implemented in this build.

**Verification (manual):**
1. `oflm image flux2-klein:4b "a red fox in fresh snow" --size 512 --seed 1 -o fox.png`:
   a coherent fox; the path, `Seed 1` and a timing line are printed.
2. The same with `-o fox.jpg`: a JPEG of the same image.
3. Each of these fails with the named reason and loads nothing:
   `-o fox.webp`; `--size 768` (supported: 512, 1024); `oflm image llama3.2:1b "x"`
   (not an image model); `oflm run llama3.2:1b --seed 1`; `oflm pull llama3.2:1b "x"`;
   `oflm run llama3.2:1b --image a.png`; `--image` of a BMP, of a 40x40 PNG, of a missing file.
4. `oflm run flux2-klein:4b` is refused as not a chat model.
5. `oflm image flux2-klein:4b "Make it a watercolor painting" --image coffee.jpg --seed 7`,
   with a phone-orientation JPEG (EXIF 6, 600x400 stored): the CLI prints "reference
   400x600 (EXIF orientation 6 applied) centre-cropped to 400x400, upscaled to 512x512" and
   writes an upright 512² watercolor of the cup (checked 2026-10-01).

### OPEN-DIFFUSION-TOKENS: oflm's prompt ids equal the pipeline's
**Applies to:** `src/open_diffusion/prompt.cpp`, `src/open_diffusion/engine.cpp`
**Verification:** test

The ids `oflm image` feeds the engine for a prompt equal `klein_pipeline.token_ids`: Qwen3's
chat template with the empty think block kept, tokenized without special tokens,
truncated to 512 after templating. The engine pads them with 151643 and masks keys past
their length. A prompt may itself contain the pad token; the length is the ids' count,
never the first pad.

**Acceptance criteria** (`specs/open-diffusion/tests/token_goldens.json`, written by
`utilities/dit-chain/klein_tokens.py --goldens`):
- The 8 study prompts give the goldens' ids exactly.
- A prompt with accents, CJK and an emoji gives the goldens' 32 ids.
- `"a prompt that names <|endoftext|> inside it"` gives 20 ids with 151643 inside them.
- The empty prompt gives the template's 12 ids.
- A prompt of 400 words gives exactly 512 ids, the goldens' first 512.

Run by the `open_diffusion_tokens` CTest (`src/open_diffusion/tokens_test.cpp`) against
the installed model's `tokenizer.json`; it fails, naming the path, without it.

### OPEN-DIFFUSION-DETERMINISM: same inputs, same bytes
**Applies to:** `src/open_diffusion`, `oflm image`
**Verification:** test

The same prompt, size and seed give the same file bytes, run to run, and the same pixels
as the pyxrt runner.

**Acceptance criteria:**
- Two `oflm image flux2-klein:4b "a red fox in fresh snow" --size 512 --seed 1` runs
  write identical PNG files (`specs/open-diffusion/tests/test_cli_determinism.py`; needs
  the NPU, `oflm.exe` and the installed model, and says which is missing when skipped).
- The native engine's pixels for study prompts 0 and 1 at 512², given their own token ids
  and the study's noise, equal `generate.py`'s
  (`specs/open-diffusion/tests/test_engine_matches_pyxrt.py`). The engine runs one
  register-configured context and the runner six xclbin contexts, so this catches a wrong
  reconfiguration, which would only drift the image. The test needs the NPU, the engine
  CLI, a built kernel directory and the study inputs, and says which is missing when
  skipped.
- Edits: two `oflm image ... --image ref.png --seed 1` runs write identical PNG files, which
  differ from the reference (`test_cli_determinism.py`). The engine's pixels for edit study
  0 at 512² equal `generate.py --edit`'s, encoder included (`test_engine_matches_pyxrt.py`;
  the edit study inputs come from `utilities/dit-ref/capture_edit_goldens.py`). Both skip,
  naming why, when the model or kernels have no 512 edit configuration.
- Packing is reproducible: the VAE attention's rank-128 factorization fixes each singular
  pair's sign, so numpy 2.4 and 2.5 pack identical weights. A sign flip once moved pixels by
  up to 10 levels: bfp16's int8 mantissa is not sign-symmetric.

### OPEN-DIFFUSION-PACKAGE: the model and its kernels are built and found with no manual step
**Applies to:** `utilities/q4nx-build` (`--open-diffusion`), `open_kernels/export_dit_kernels.py` (`--install`), `src/open_diffusion`, `src/model_list.json`, `src/model_info.json`
**Verification:** manual

- `q4nx-build --open-diffusion -i black-forest-labs/FLUX.2-klein-4B -o <dir>` builds the
  whole model directory from the checkpoint (flat: 17 files, ~9 GB) and
  `model_info_entry.json`. It refuses a checkpoint whose pipeline class, transformer or
  text-encoder geometry is not klein 4B's, naming the fields.
- `export_dit_kernels.py` builds the six kernel sets, then assembles them into one full
  ELF per resolution (`open_kernels/compose_elf.py`, `diffusion_r<R>.elf`).
  `--install <dir>` copies only the ELFs and their description, removes an earlier
  install's kernel-set files, and writes `diffusion_kernels.json` (format
  `oflm-open-diffusion-kernels-v2`) last. It refuses a directory built from other stream
  specs than the tree's, or whose ELFs are missing or older than its sets.
- The model directory and the kernel set carry the same layout hash (every stream spec
  and the weight packing). The engine refuses a kernel set whose manifest is missing,
  incomplete, of another format (a v1 set of six xclbins included) or of another layout,
  or whose ELFs are missing, naming which.
- The engine finds its kernels in this order: `OFLM_DIFFUSION_KERNELS_DIR` (used as
  given), `<model dir>/open_kernels`, then `<root>/xclbins/FLUX.2-klein-4B-NPU2/open_kernels`
  for each xclbins root. The installer ships the last.
- `oflm pull flux2-klein:4b` installs the model from `Cyronius/FLUX.2-klein-4B-NPU2`,
  verifying every file's hash.

**Verification (manual):**
1. Build the model directory into an empty models root with `q4nx-build --open-diffusion`;
   `oflm list` shows `flux2-klein:4b` installed.
2. `oflm image` finds the installed kernel set with no variable set (it prints "an xclbins
   root").
3. Point `OFLM_DIFFUSION_KERNELS_DIR` at a copy of the set whose manifest layout is
   edited: refused, naming both hashes.
4. On a machine without the model: `oflm pull flux2-klein:4b` downloads 17 files and
   reports them verified.

**Verified 2026-09-28:** built from `black-forest-labs/FLUX.2-klein-4B` (layout
`e0450140c44f78e0`) and uploaded to `Cyronius/FLUX.2-klein-4B-NPU2`. The live tree
listing matched the builder's predicted registry entry for all 17 files. `oflm pull`
into an empty models root verified all 17 hashes, and the pulled model's 512² seed-1 fox
has the same bytes as the locally built one. `oflm image` found the shipped kernel set
with nothing set ("an xclbins root"). A manifest with an edited layout was refused,
naming both hashes.

### OPEN-DIFFUSION-EDIT: an edit follows its prompt and keeps its reference
**Applies to:** `open_kernels/klein_pipeline.py` (`plan(R, edit=True)`), `open_kernels/vae_encoder.py`, `src/open_diffusion`, `oflm image --image`, `/v1/images/edits`
**Verification:** manual

klein's edit, as diffusers' `Flux2KleinPipeline(image=...)` runs it (diffusers 0.40):
- the reference's VAE latents (the encoder's mean), patchified 2x2 and BN-normalised, are
  T extra tokens after the generated ones;
- their positions are (10, h, w, 0);
- they get the generated tokens' timestep modulation in every block, and are never read
  back;
- the sigmas count the generated tokens only;
- no mask, no strength, no guidance.

One reference; a mask or a second reference is refused, naming it as not implemented.

The edit study (`capture_edit_goldens.py`) has 12 references: the 8 study images, plus 4
public-domain / CC0 photos, among them an EXIF-6 JPEG and a grayscale scan. Each comes with
an edit prompt and fixed noise. The NPU's edits must be coherent, and must:
- follow the prompt;
- keep the reference's scene;
- land near diffusers' bf16 edits in LPIPS;
- **ablation:** with each edit given another study reference (`generate.py --swap-ref 1`),
  LPIPS against diffusers' edit must be far higher. A DiT that ignored the reference would
  still make coherent images, so only this catches it.

**Verification (manual):**

```
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_edit_goldens.py --size 512
python utilities\dit-chain\generate.py --kernels <set> --size 512 --edit --study C:\dev\ditref-out\klein_edit_512_s4 --prompts 12 --out <dir> [--swap-ref 1 | --ref-tokens]
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_images.py <dir> --test "{:02d}.png" --ref "{:02d}.png" --ref-dir C:\dev\ditref-out\klein_edit_512_s4\bf16 --n 12
```

**Measured 2026-10-01**, 512², 12 edits, LPIPS vs diffusers bf16:

| run | LPIPS mean / max | PSNR |
|---|---|---:|
| **everything on the NPU** | **0.0133 / 0.041** | 33.1 dB |
| the same with diffusers' bf16 reference tokens (`--ref-tokens`) | 0.0140 / 0.048 | 32.5 dB |
| diffusers bf16, with the NPU encoder's tokens (CPU) | 0.0034 / 0.009 | 39.5 dB |
| **ablation: each edit with the next edit's reference** | **0.73 / 0.85** | 9.5 dB |
| 1024², 2 edits (study prompts 0-1), everything on the NPU | 0.0072 / 0.0088 | 36.4 dB |
| 1024², the ablation | 0.84 / 0.85 | 8.2 dB |

- All 12 are coherent and follow their prompts: OPEN → CLOSED, TOMATO → PUMPKIN (legible),
  the red fox made arctic, the coffee photo's espresso made latte art, upright.
- The drift is the DiT's and the text encoder's, not the encoder's: swapping in diffusers'
  reference tokens changes nothing measurable.
- Edits drift far less than generations (0.107), because the reference anchors them.
- The CPU emulation's prediction (`capture_edit_goldens.py --variants npu-emul`) is not
  measured yet.

### OPEN-DIFFUSION-ENCODER: the NPU encoder's reference tokens decode like diffusers'
**Applies to:** `open_kernels/vae_encoder.py`, `open_kernels/designs/vae_ew` (`npix`)
**Verification:** manual

The encoder runs as the decoder's kernels do. Where it needs more, it uses existing
kernels with new layouts:
- the three stride-2 downsamples run as space-to-depth: the residual add before each is 4
  phase dispatches, then a 3x3 conv with zero -1 taps (`utilities/dit-chain/spike_s2d.py`);
- the mid-block attention is factored at rank 128, like the decoder's;
- conv_out, quant_conv's mean, the 2x2 patchify and the BN normalisation are one conv on
  norm_out's space-to-depth grid.

The tokens must decode, in fp32, to images close to the decode of diffusers' fp32 encoder
mean. They are not compared token for token: the decoder ignores much of their error, and
the DiT barely sees it (OPEN-DIFFUSION-EDIT).

**Verification (manual):**

```
python utilities\dit-chain\chain_test_vae_enc.py --kernels <set root> [--build] --goldens C:\dev\ditref-out\klein_edit_512_s4
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_ref_latents.py C:\dev\ditref-out\klein_edit_512_s4
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\encoder_study.py C:\dev\ditref-out\klein_edit_512_s4 --npu-dump <dir>
```

**Measured 2026-10-01**, 512², the 12 study references (1024², 2 references: decoded LPIPS
0.0013 / 0.0014, PSNR 47.7 dB, tokens 11.3%):
- Decoded LPIPS 0.0008 mean, 0.0010 max, PSNR 49.1 dB. Diffusers' bf16 encode: 0.0000,
  63.3 dB.
- Token rel_fro against the fp32 mean: 11.7% mean, 28% max. Diffusers' bf16: 1.0%.
- `encoder_study.py` splits the rel_fro, with both factors also in the decoder:
  - the rank-128 attention: 2.6% → 9.5% in emulation;
  - dit_conv's bf16 accumulator re-rounding every 64 of K, which the emulation leaves out.
- 89 dispatches, ~0.35-0.45 s in the engine.

### OPEN-DIFFUSION-REFERENCE: the reference is prepared as diffusers' PIL path prepares it
**Applies to:** `src/open_diffusion/reference.cpp` (`oflm image --image`, `/v1/images/edits`, `open_diffusion_cli --ref`)
**Verification:** test
**External tests:** none (`specs/open-diffusion/tests/test_reference.py`; the tool comes from `src/open_diffusion/build.cmd`)

An edit's reference file becomes the R x R RGB8 image the encoder reads, as
`diffusers.utils.load_image` and PIL would make it:
- PNG and JPEG only (stb_image);
- a JPEG's EXIF orientation applied, as `ImageOps.exif_transpose`;
- RGB, alpha dropped and gray expanded, as `convert("RGB")`;
- centre-cropped to a square;
- resized to R with a port of PIL's LANCZOS: its coefficients, its two uint8 passes, its
  truncated edges.

Square-cropping deviates from diffusers, which keeps the aspect ratio: a non-square
reference loses its sides, and the edit is square. A reference smaller than R is
upscaled, where diffusers never upscales. Aspect buckets (an encoder per bucket, plus
dit_fa's valid_len) are the path past this (`plans/edits.md`, decision 2).

Refused, naming the reason, before decoding where the header shows it: a format other
than PNG or JPEG (as not implemented); a side under 64 px; an aspect over 8:1; over
64 megapixels; an undecodable file.

**Acceptance criteria:**
- A 512² PNG at R = 512 comes back byte for byte.
- PNG crops resized to R are within 1 level of PIL's crop + LANCZOS, at every pixel:
  4000x3000 and 640x480 to 512, 3000x4000 and 1000x750 to 1024, 300x300 up to 512.
- A 4000x3000 JPEG with magenta bars outside its centred 3000x3000 square shows no
  magenta. It is within JPEG decoding's differences of PIL (stb_image vs libjpeg): mean
  under 0.5 levels, 99.9th percentile within 3.
- RGBA, LA, L and P PNGs come back equal to PIL's `convert("RGB")`.
- Each EXIF orientation 1-8 equals `ImageOps.exif_transpose` of the same decoded pixels.
  Orientation 6 end to end is within JPEG decoding's differences of PIL's path.
- 63x100, 100x63 and 900x100 PNGs, a 9000x8000 PNG, BMP/WebP/GIF files and a truncated
  PNG are refused, each naming its reason.

### OPEN-DIFFUSION-PERF: what may be called a performance number
**Applies to:** the whole pipeline
**Verification:** manual

A time quoted for this pipeline must meet all of these:
- it is a warm generation, not the first after load;
- the NPU is in turbo mode;
- nothing else is loading the CPU. The CPU shares the package power budget, and a busy
  CPU slows the NPU 1.2-1.6×.

A number measured under load says so.

**Measured 2026-09-29, native engine, one hardware context per resolution**
(`specs/open-diffusion/archive/one-context.md`). Turbo; light load (CPU 11-23%
before each run: chat and editor apps, nothing computing). The study's prompt 0 with its
own 29 token ids and the study's noise; 2 processes × 3 warm runs each:

| | 512² | 1024² |
|---|---:|---:|
| image, warm | **3.66-3.69 s** | **11.98-12.02 s** |
| text encoder | 0.32 s | 0.32 s |
| conditioning | 0.03-0.04 s | 0.03-0.04 s |
| one denoising step | 0.75 s | 2.61-2.65 s |
| VAE | 0.32-0.33 s | 1.09-1.13 s |
| first image after load | 3.68-3.69 s | 12.02 s |
| load (cached) | 5.5-5.7 s | 5.7-5.9 s |
| host CPU per image | 0.06-0.20 s | 0.11-0.22 s |

The six-context engine (the one below, rebuilt from `feat/images-api`) was interleaved
with it under the same conditions: 5.14-5.31 s at 512² and 13.42-13.48 s at 1024² in its
least-loaded rounds, loading in 5.7-5.9 s. The one-context engine is 28% faster at 512²
and 11% at 1024². What it removes is ~822 kernel-set switches of ~2.1 ms each, now
configurations of 0.3-0.7 ms. The pixels are the same bytes
(OPEN-DIFFUSION-DETERMINISM).

**Re-measured 2026-09-30, quiet** (CPU 1-16% before each run, nothing computing), same
binaries and inputs, 2 processes × 4 runs: 512² 3.75-3.96 s (3.75-3.78 s warm), text
encoder 0.32-0.35 s, step 0.76-0.84 s, VAE 0.32-0.35 s; 1024² 12.16-12.40 s, step
2.65-2.74 s, VAE 1.12-1.20 s; load 5.1-5.8 s. That is no faster than the light-load
figures above, which stand as the range. Logs: `C:\dev\switch-work\quiet-0930`.

**Contention:** another process using the NPU during an image
(`utilities/reconfig-probe/contention_trial.ps1`, a six-xclbin-context contender).
- At the engine's high QoS priority: 6 of 6 trials (24 images, 3 of them 1024²) were
  byte-identical, and the contender kept running. Images took ~10-25% longer while it
  ran.
- At normal priority every variant hung: 8 of 8 trials (`ERT_CMD_STATE_TIMEOUT`), usually
  taking the contender down too.
- Not tested: a contender that itself runs at high or realtime priority (Windows Studio
  Effects may).

**Measured 2026-09-27, native engine (six xclbin contexts), quiet machine** (CPU load 3-5% before each run),
turbo, the study's prompts 0-1 and noise (`goldens_pipe_<R>\ids_i.npy`, `noise_i.npy`),
4 runs each:

| | 512² | 1024² |
|---|---:|---:|
| image, warm | 5.09-5.15 s | 13.36-13.42 s |
| text encoder | 0.64-0.65 s | 0.64-0.65 s |
| conditioning | 0.04 s | 0.04 s |
| one denoising step | 0.98-1.00 s | 2.85-2.88 s |
| VAE | 0.46 s | 1.22-1.28 s |
| first image after load | 5.12-5.14 s | 13.42 s |
| load (cached) | 5.3-5.4 s | 5.2-5.4 s |
| host CPU per image | 0.00-0.11 s | 0.00-0.06 s |

LPIPS against the bf16 CPU run: 0.19 / 0.10 (512²) and 0.33 / 0.15 (1024²), the images
recorded under OPEN-DIFFUSION-QUALITY.

**The same machine's iGPU, for comparison** (Radeon 890M, gfx1150; measured 2026-09-27
on the quiet machine, right before the NPU run above). This is the diffusers bf16
pipeline through PyTorch-ROCm (`torch 2.12.0+rocm7.14.1`), timed by
`utilities/dit-ref/igpu_bench.py`:

| | 512² | 1024² |
|---|---:|---:|
| image, warm | 20.05-20.45 s (6 runs) | 134.2 s (1 run) |
| text encoder | 1.30-1.36 s | 1.31 s |
| one denoising step | 4.46-4.53 s | 31.9 s |
| VAE | 0.66-0.71 s | 4.75 s |
| first image after load | 52 s | 193 s |
| host CPU per image | 21.7-22.7 s | 136 s |

- The NPU is 3.9× faster at 512² and 10× faster at 1024².
- The iGPU's LPIPS against the bf16 CPU run is 0.069 / 0.015 (512²). It is plain bf16,
  so it drifts less than the NPU.
- The iGPU has no fused attention on this build:
  - by default, flash and memory-efficient SDPA have no gfx1150 kernel;
  - with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, they fail to launch (`invalid
    argument`).
  - Attention therefore runs on the math path at ~0.5 TFLOPS. That is 1.35 s of each
    512² step and 14-17 s of each 1024² step.
- A bf16 4096³ GEMM on the iGPU runs 8.5 TFLOPS. At that rate for every FLOP, a step would
  take ~1.5 s at 512², still slower than the NPU's 1.0 s.
- The iGPU's host CPU time about equals its wall time. The ROCm runtime keeps a core
  busy for the whole image.

**Measured 2026-09-27, native engine, under load** (`src/open_diffusion`), turbo, 2
prompts × 3 warm runs each. **Another ~11-core CPU job (another session's) was
running**, so these are pessimistic:

| | 512² | 1024² |
|---|---:|---:|
| image, warm | 5.2-6.0 s (mean 5.6) | 13.5-15.9 s (mean 15.1) |
| text encoder | 0.65-0.80 s | 0.64-0.75 s |
| conditioning | 0.04-0.06 s | 0.04-0.06 s |
| one denoising step | 1.0-1.25 s | 2.9-3.5 s |
| VAE | 0.46-0.54 s | 1.27-1.46 s |
| first image after load | 5.7-6.3 s | 15.0-15.8 s |
| load (kernels + 7.5 GB weights, cached) | 8-9 s | 6-7 s |
| host CPU per image | 0.03-0.19 s | 0.03-0.11 s |

**Earlier the same day, pyxrt runner** (`generate.py`), with a different 12-core job
running. These are slower: Python dispatch, and more contention. They were pessimistic
too:
- the standalone attention benchmark ran 42.7 ms against its quiet 30.5 ms;
- `sgl_in` ran 68.7 ms against 59.4 ms.

| | 512² | 1024² |
|---|---:|---:|
| image, warm | 6.5-7.4 s | 20.0-20.9 s |
| text encoder | 0.76-0.89 s | 0.86-0.88 s |
| conditioning | 0.06 s | 0.07 s |
| 4 denoising steps | 5.1-5.8 s | 17.2-17.9 s |
| VAE | 0.60-0.67 s | 1.86-2.07 s |
| first image after load | 9.5-11 s | 29 s |
