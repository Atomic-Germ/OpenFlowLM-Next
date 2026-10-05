# open_diffusion: FLUX.2 [klein] 4B text-to-image and edits on the NPU

A native engine that replays `open_kernels/klein_pipeline.py`'s schedule. The schedule
has 1050 dispatches over six kernel sets:
- the text encoder;
- conditioning;
- 4 denoising steps;
- the VAE.

One `Engine` holds every resolution the bundle has: the weights load once, and
`select(size, steps)` opens a resolution's context and allocates its activations the
first time. The constructor that takes a size opens that one while the weights load. Any step
count from 1 to 50 runs: step k is the bundle's step 0 with its modulation and dt views
moved on by their strides, and a count other than the bundle's gets its sigmas from
`schedule.hpp` (OPEN-DIFFUSION-STEPS).

An edit (`select(size, steps, true)`, then `set_reference`) is a configuration of its own,
`<R>e<R>`, with its own ELF: the VAE encoder runs as an `encode` phase before the steps, and the
reference's tokens follow the generated ones in every block (1143 dispatches at 512²).
`reference.hpp` prepares the reference from file bytes: PNG or JPEG, EXIF orientation, a
centre crop and PIL's LANCZOS (OPEN-DIFFUSION-REFERENCE). `open_diffusion_cli --ref` takes
a file or a prepared `.npy`.

Every op runs on the NPU. Per image, the host does only these things:
- writes the prompt's 512 embedding rows and the noise (and an edit's reference);
- picks `te_attn`'s `valid_len` head for the prompt's length;
- reads the RGBA and encodes the PNG or JPEG (`stb_image_write`, vendored in
  `third_party/stb`).

A resolution runs in ONE hardware context: its six kernel sets are devices of one full
ELF (`open_kernels/compose_elf.py`). A set change is a configure-only kernel,
`main:cfg_<set>`: register writes costing 0.3-0.7 ms, where switching between six xclbin
contexts cost ~2.1 ms. Each set's configure and the ops after it go to the NPU as one
`xrt::runlist`. The context runs at high QoS priority, and every configure starts from a
reset. Another process using the NPU then gets it only between those stretches, and
cannot leave ours half-configured (at normal priority it did, and both processes hung).
The host blocks (XRT's wait sleeps) only at phase boundaries and when 32 stretches are in
flight.

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

Every build has the interface: `oflm image` and the server ask `open_diffusion::available()`,
never which runtime was built. The engine drives XRT directly (a full-ELF context per
resolution, sub-buffer views); an HRX build compiles `engine_unavailable.cpp` instead, which answers
"not implemented in this build".

## Build and run

```
oflm image flux2-klein:4b "a red fox in fresh snow" --seed 1 -o fox.png
```

The standalone gate, for kernel and schedule work:

```
. C:\dev\mlir-aie\iron_env.ps1
python open_kernels\export_dit_kernels.py --resolutions 512,1024 --out C:\dev\klein-kernels   # the sets, then their ELFs
python open_kernels\export_dit_kernels.py --out C:\dev\klein-kernels --install src\xclbins\FLUX.2-klein-4B-NPU2\open_kernels
python utilities\dit-chain\export_bundle.py --out C:\dev\klein-model --pack-cache C:\dev\klein-kernels\packed
src\open_diffusion\build.cmd
python utilities\dit-chain\klein_tokens.py "a red fox in fresh snow" C:\dev\fox.npy --bundle C:\dev\klein-model
src\open_diffusion\out\open_diffusion_cli.exe --model C:\dev\klein-model --kernels src\xclbins\FLUX.2-klein-4B-NPU2\open_kernels --size 1024 --ids C:\dev\fox.npy --seed 1 --out fox.png
```

`--noise <npy>` injects packed initial latents (bf16 bits). `capture_pipeline_inputs.py`
writes the study's. `--ids` takes the prompt's own tokens, unpadded: the study's
`ids_<i>.npy` are padded to 512, and passed whole they run with no text mask. `--steps N` changes the step count; `--runs N` repeats the image;
`--profile` times each op.

## Not implemented yet

- **Inpainting and multi-reference edits.** `/v1/images/edits` takes one reference and no
  mask (`specs/open-diffusion/plans/edits.md`); a mask or a second image is a 400 naming it.
- **Non-square references.** They are centre-cropped to a square (OPEN-DIFFUSION-REFERENCE).
- **`oflm add` for klein derivatives.** `oflm-add` requires `model.q4nx`.
