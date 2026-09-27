# open_diffusion: FLUX.2 [klein] 4B text-to-image on the NPU

A native engine that replays `open_kernels/klein_pipeline.py`'s schedule. The schedule
has 1050 dispatches over six kernel sets:
- the text encoder;
- conditioning;
- 4 denoising steps;
- the VAE.

Every op runs on the NPU. Per image, the host does only these things:
- writes the prompt's 512 embedding rows and the noise;
- patches `te_attn`'s `valid_len`;
- reads the RGBA and writes the PNG.

Runs on one hardware context are queued back to back. Before the next kernel set, the
host blocks on every queued run; XRT's wait sleeps.

Its output is pixel-identical to the pyxrt runner (`utilities/dit-chain/generate.py`)
for the same token ids and noise.

Spec: `specs/open-diffusion/spec.md`. Design: `.claude/plans/image-diffusion-phase6-engine.md`.

## Build and run

```
. C:\dev\mlir-aie\iron_env.ps1
python open_kernels\export_dit_kernels.py --resolutions 512,1024 --out C:\dev\klein-kernels
python utilities\dit-chain\generate.py --kernels C:\dev\klein-kernels --pack-only       # ~8 GB, once
python utilities\dit-chain\export_bundle.py --kernels C:\dev\klein-kernels --out C:\dev\klein-bundle
src\open_diffusion\build.cmd
python utilities\dit-chain\klein_tokens.py "a red fox in fresh snow" C:\dev\fox.npy
src\open_diffusion\out\open_diffusion_cli.exe --bundle C:\dev\klein-bundle --size 1024 --ids C:\dev\fox.npy --seed 1 --out fox.png
```

`--noise <npy>` injects packed initial latents (bf16 bits). `capture_pipeline_inputs.py`
writes the study's. `--runs N` repeats the image; `--profile` times each op.

## Not implemented yet

- **Tokenizing.** The engine takes token ids. `src/common/tokenizer` (tokenizers-cpp) does
  it in the main `flm` build, and the standalone build has no Qwen tokenizer. For now
  `utilities/dit-chain/klein_tokens.py` writes the ids.
- **Serving.** `oflm serve`'s `/v1/images/generations` is a separate plan.
- **Packaging.** `q4nx-build --open-diffusion` and `oflm add` are separate work. The bundle
  is a directory whose weights are referenced from `<kernels>\packed`.
