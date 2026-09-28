---
name: open-diffusion
description: Build, run, verify and time FLUX.2 [klein] 4B text-to-image with every op on the XDNA2 NPU -- the whole-image schedule (open_kernels/klein_pipeline.py), the pyxrt runner (utilities/dit-chain/generate.py), the bundle exporter and the native engine (src/open_diffusion). Use when generating images on the NPU, changing the schedule or its buffer layout, adding a resolution or another klein-shaped model, checking image quality against the CPU reference, timing the pipeline, or debugging a hang / wrong image in it.
---

# FLUX.2 [klein] on the NPU: the whole image

Sources:
- `open_kernels/klein_pipeline.py`: the ONE schedule, which everything else follows:
  - buffers, and the ops in dispatch order (1050);
  - weight packing specs, the parameter table and the host setup.
- `open_kernels/export_dit_kernels.py`: builds every stream the schedule names.
- `utilities/dit-chain/generate.py`: the pyxrt runner.
- `utilities/dit-chain/export_bundle.py`: the model directory (schedules, packed weights in
  one `weights.bin`, the embedding table), for `src/open_diffusion` (native,
  pixel-identical to generate.py). `q4nx-build --open-diffusion` runs it.
- `src/open_diffusion/`: the engine; `prompt.cpp` (main build) templates and tokenizes;
  `src/src/image_command.hpp` is `oflm image`; `cli.cpp` is the standalone gate.
- Serving: `/v1/images/generations` and `/v1/images/edits` (501) in
  `src/server/rest_handler.cpp`; the request rules are `openai_compat::images_request()`.
  Spec: `specs/server-api/spec.md` SERVER-IMAGES-*; tests
  `specs/server-api/tests/test_images_api.py` (needs `oflm serve <chat model>`).
- The kernels have their own skills: `dit-gemm`, `dit-fa`, `dit-ew`, `dit-conv`.
- Spec: `specs/open-diffusion/spec.md`. Design: `.claude/plans/image-diffusion-phase6-engine.md`.

## Run it

```
. C:\dev\mlir-aie\iron_env.ps1
python open_kernels\export_dit_kernels.py --resolutions 512,1024 --out C:\dev\klein-kernels --jobs 6
python open_kernels\export_dit_kernels.py --out C:\dev\klein-kernels --install src\xclbins\FLUX.2-klein-4B-NPU2\open_kernels
python utilities\dit-chain\generate.py --kernels C:\dev\klein-kernels --pack-only          # ~8 GB, 5 min, once
python utilities\dit-chain\generate.py --kernels C:\dev\klein-kernels --size 512 --prompt "a red fox in snow" --out C:\dev\gen
```

The shipped path (what a user gets from `oflm image`):

```
cd utilities\q4nx-build
python -c "from q4nx.cli import main; main(['--open-diffusion','-i','black-forest-labs/FLUX.2-klein-4B','-o','%OFLM_MODEL_PATH%\models\FLUX.2-klein-4B-NPU2','--pack-cache','C:\dev\klein-kernels\packed'])"
src\build-windows-vcpkg.cmd
src\out\oflm.exe image flux2-klein:4b "a red fox in fresh snow" -o fox.png
src\open_diffusion\build.cmd     # the standalone gate: --model <dir> --kernels <installed set> --ids ids.npy
```

- **Two directories, one layout hash.** The model directory (`bundle.json`, `config.json`)
  and the installed kernel set (`diffusion_kernels.json`) each carry
  `export_dit_kernels.layout_hash`: every stream spec plus `WEIGHT_FORMAT`. The engine
  refuses a mismatch by name. Change a stream spec or `pack_b` and both must be rebuilt;
  bump `WEIGHT_FORMAT` when only the packing changes.
- **Kernel search:** `OFLM_DIFFUSION_KERNELS_DIR`, then `<model>/open_kernels`, then
  `<root>/xclbins/FLUX.2-klein-4B-NPU2/open_kernels` per xclbins root (`find_kernels`).
- **`--install` copies only runtime files** (157, 15 MB). Never ship `--out` itself: the
  installer copies `xclbins\` recursively, `build\` included.
- **The registry entry:** `model_info_entry.json` predicts the HF tree API's listing
  (`*.bin` and >= 10 MB files are LFS). After the upload to `Cyronius/FLUX.2-klein-4B-NPU2`,
  pull on a clean models directory: every file's hash is checked.
- **Tokens:** `specs/open-diffusion/tests/token_goldens.json` (from `klein_tokens.py
  --goldens`) pins oflm's prompt path to `klein_pipeline.token_ids`; the
  `open_diffusion_tokens` CTest runs it. A prompt may contain `<|endoftext|>` (the pad id):
  the engine takes the ids' length as given, never scans for the pad.

Run PowerShell scripts that take `--out` through a wrapper with no `param()` block. With
`[Parameter(ValueFromRemainingArguments)]`, `--out` binds to `-OutVariable`.

## Quality check (the gate)

```
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_pipeline_inputs.py --size 512    # ids, noise, sched, temb, mod
python utilities\dit-chain\generate.py --kernels C:\dev\klein-kernels --size 512 --study C:\dev\ditref-out\goldens_pipe_512 --out <d> [--ctx-ref] --ref-latents C:\dev\ditref-out\goldens_vae_512
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_images.py <d> --test "{:02d}.png" --ref "{:02d}.png" --ref-dir C:\dev\ditref-out\klein_512_s4\bf16
```

Expect LPIPS ~0.107 for the full pipeline and ~0.044 with `--ctx-ref`. Look at the
images.
- Prompt 0 (the neon sign) and prompt 3 are chaotic: they swing 2× under any
  perturbation, so judge on the mean and on the images.
- Final-latent rel_fro against the bf16 run is 0.2-0.6. That is trajectory divergence,
  not a bug.

## What was learned getting here (don't re-derive it)

1. **Modulation order is fixed by one GEMM's columns.** All steps' modulation comes from
   one GEMM (M = steps padded to 512, N = 184,320).
   - Each run of 2-3 vectors a dit_ew op reads has its own 6-vector, 4096-aligned slot
     (`MOD_RUNS`), so a step's run is a sub-buffer view.
   - The last double block's res modulates for the single blocks.
   - The last single block's res applies norm_out, whose chunk order is (scale, shift).
2. **te_attn's valid_len is an RTP write.** The exporter diffs a probe build (`valid_len
   77`) and records the 32 words, one per core, in `fa/dit_fa.json` `patch`. Runners
   rewrite them per prompt.
3. **The text-encoder taps (hidden states 9/18/27) are a second res+RMSNorm pass**
   (`te_tap<k>`). Its residual output goes to CTX [512, 8192] at column 2560 k.
   - The 512 zero columns spill into the next slot, or past 7680.
   - ctx_emb reads CTX with lda 8192.
4. **Conditioning GEMMs cost ~0.009 LPIPS.** dit_gemm's bf16 accumulator puts the
   modulation vectors 2-3% from diffusers'. Check with `utilities/dit-chain` +
   `goldens_pipe_<R>/mod.npz`: read MOD after a cond-only run. A layout bug would be
   O(1).
5. **Every queued xrt::run must be waited on**, not just the last. A run never waited on
   keeps a stale state, and restarting it fails with "bad command state". Queued runs
   across two hardware contexts hang the array, so drain before every switch.
6. **Determinism is a check.** The same inputs must give the same pixels, run after run and
   Python vs native.
   - Nondeterminism means state carried in a buffer. Zero-bordered VAE buffers were one
     case (`dit-conv` skill, lesson 11).
   - Diff the final latents first to localize it.
7. **Six hardware contexts in one process work** (gemm, fa, ew, conv, conv1, vew). About
   960 set switches per image at ~2 ms each are ~2 s at either size, the biggest
   fixed cost at 512².
8. **Timing is invalid while any CPU job runs.** The CPU shares the package power budget.
   A 12-core job made attention 42.7 ms against its quiet 30.5 ms.
   - Check `Get-Process` before quoting a number.
   - Keep `xrt-smi` in turbo.

## Serving (what was learned)

- **Swap by default, `--image 1` to keep both.** An image request resets `auto_chat_engine`
  but keeps `current_model_tag`, so an omitted or same-named chat model reloads through
  `ensure_model_loaded`'s NeedsLoad path. `release_image_engine_for_chat()` is the other
  direction. llama3.2:1b plus the image engine fit the NPU together; with `--asr`/`--embed`
  too is not measured.
- **The engine takes the server's `xrt::device`** (`Engine(model, kernels, &npu_device_inst)`).
- **`XT` is overwritten by the text encoder** (the residual stream), so `set_tokens` runs
  before every image, not once per request.
- **Step counts:** the engine derives each arg's per-step stride from step 0 vs step 1 of
  the schedule (only `MOD` and `DT` move) and grows `DT` to fit 50 steps. For the bundle's
  count it writes the bundle's own `tf_/dt_` bytes: `schedule.hpp`'s features are within
  2^-8, not equal, because numpy's float32 `exp` is not correctly rounded.
- `open_diffusion_schedule` CTest holds `schedule.hpp` to the bundle; `openai_compat` holds
  `images_request()`.

## Adding a resolution

- `check_resolution`: (R/16)² must be a multiple of 512.
- Export with `--resolutions`, `--install` it, then rebuild the model directory (the
  layout hash changes) and add the size to `image_sizes` in `src/model_list.json`.
- New streams appear per R: DiT gemm/ew/fa, euler, x_emb/proj_out, and the VAE's.
- Capture study inputs with `capture_pipeline_inputs.py --size R`.
