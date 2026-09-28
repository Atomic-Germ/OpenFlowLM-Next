# open_diffusion: FLUX.2 [klein] 4B text-to-image on the NPU

A native engine that replays `open_kernels/klein_pipeline.py`'s schedule. The schedule
has 1050 dispatches over six kernel sets:
- the text encoder;
- conditioning;
- 4 denoising steps;
- the VAE.

One `Engine` holds every resolution the bundle has: the weights and kernel sets load once,
and `select(size, steps)` allocates a resolution's activations the first time. Any step
count from 1 to 50 runs: step k is the bundle's step 0 with its modulation and dt views
moved on by their strides, and a count other than the bundle's gets its sigmas from
`schedule.hpp` (OPEN-DIFFUSION-STEPS).

Every op runs on the NPU. Per image, the host does only these things:
- writes the prompt's 512 embedding rows and the noise;
- patches `te_attn`'s `valid_len`;
- reads the RGBA and encodes the PNG or JPEG (`stb_image_write`, vendored in
  `third_party/stb`).

Runs on one hardware context are queued back to back. Before the next kernel set, the
host blocks on every queued run; XRT's wait sleeps.

Its output is pixel-identical to the pyxrt runner (`utilities/dit-chain/generate.py`)
for the same token ids and noise.

Spec: `specs/open-diffusion/spec.md`. Design: `.claude/plans/image-diffusion-phase6-engine.md`.

It loads two directories, which must carry the same layout hash:
- the model: `q4nx-build --open-diffusion` (`utilities/dit-chain/export_bundle.py`),
  installed as `flux2-klein:4b`;
- a kernel set: `export_dit_kernels.py --install`, found by `find_kernels`
  (`OFLM_DIFFUSION_KERNELS_DIR`, `<model>/open_kernels`, then the xclbins roots).

`oflm image` (`src/src/image_command.hpp`) and `oflm serve`'s `/v1/images/generations`
(`src/server/rest_handler.cpp`, `specs/server-api/spec.md` SERVER-IMAGES-*) are the
user-facing paths; `prompt.cpp` templates and tokenizes there. `cli.cpp` is the standalone
gate, outside the main build.

## Build and run

```
oflm image flux2-klein:4b "a red fox in fresh snow" --seed 1 -o fox.png
```

The standalone gate, for kernel and schedule work:

```
. C:\dev\mlir-aie\iron_env.ps1
python open_kernels\export_dit_kernels.py --resolutions 512,1024 --out C:\dev\klein-kernels
python open_kernels\export_dit_kernels.py --out C:\dev\klein-kernels --install src\xclbins\FLUX.2-klein-4B-NPU2\open_kernels
python utilities\dit-chain\export_bundle.py --out C:\dev\klein-model --pack-cache C:\dev\klein-kernels\packed
src\open_diffusion\build.cmd
python utilities\dit-chain\klein_tokens.py "a red fox in fresh snow" C:\dev\fox.npy --bundle C:\dev\klein-model
src\open_diffusion\out\open_diffusion_cli.exe --model C:\dev\klein-model --kernels src\xclbins\FLUX.2-klein-4B-NPU2\open_kernels --size 1024 --ids C:\dev\fox.npy --seed 1 --out fox.png
```

`--noise <npy>` injects packed initial latents (bf16 bits). `capture_pipeline_inputs.py`
writes the study's. `--steps N` changes the step count; `--runs N` repeats the image;
`--profile` times each op.

## Not implemented yet

- **Image edits.** `/v1/images/edits` answers 501: klein's reference-image path needs a
  VAE encoder on the NPU.
- **`oflm add` for klein derivatives.** `oflm-add` requires `model.q4nx`.
