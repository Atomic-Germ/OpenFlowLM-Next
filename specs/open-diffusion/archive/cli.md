# Plan: FLUX.2 [klein] 4B as an installable model, run by `oflm image`

Status: **done (2026-09-28).** Every step, including the upload to
`Cyronius/FLUX.2-klein-4B-NPU2` and a verified clean pull; the requirements are merged into
`spec.md`. This is the first of two PRs (the user's split, 2026-09-27). This
branch (`feat/open-diffusion`) lands the engine, usable from the `oflm` CLI. The OpenAI
Images API (`specs/server-api/plans/images-api.md`) follows as its own PR.

Today the engine runs only on the machine that built it:
- `open_diffusion_cli.exe` is a standalone build (`src/open_diffusion/build.cmd`), outside
  `oflm`.
- It takes token ids from a Python script (`utilities/dit-chain/klein_tokens.py`).
- `bundle.json` holds absolute paths:
  - the kernel sets (`C:\dev\klein-kernels`);
  - 194 weight files in `<kernels>\packed` (`engine.cpp:158`, `:183`).
- It writes an uncompressed PNG (`write_png_rgb`, stored deflate blocks): 3.1 MB at
  1024².

**Done when** a machine with only the installer and a network connection runs this, which
pulls the model and writes the image:

```
oflm image flux2-klein:4b "a red fox in fresh snow" -o fox.png
```

## Spec impact (`specs/open-diffusion/spec.md`)

New:

| ID | what | Verification |
|---|---|---|
| OPEN-DIFFUSION-CLI | `oflm image <tag> "<prompt>" [-o FILE] [--size 512\|1024] [--seed N]` writes one image and prints its path, seed and timing. The format comes from the extension (`.png`, `.jpg`/`.jpeg`); any other is refused before loading. A tag that is not an image model is refused before any download. An unsupported size is refused, naming the supported ones. `oflm run` and `oflm serve` refuse the image tag as not a chat model. | manual |
| OPEN-DIFFUSION-TOKENS | the engine's ids for a prompt equal `klein_pipeline.token_ids`: Qwen3's template with the empty think block kept, truncated to 512 after templating, padded with 151643 | test (C++, golden ids) |
| OPEN-DIFFUSION-DETERMINISM | the same prompt, size and seed give the same bytes, run to run. Measured already: yes, run to run and Python vs native | test (integration: two `oflm image` runs at 512²) |
| OPEN-DIFFUSION-PACKAGE | `q4nx-build --open-diffusion` builds the model directory from the Hugging Face checkpoint with no manual step. `oflm pull flux2-klein:4b` installs it. The engine finds its kernel sets by the search order below and refuses a set whose format does not match the model's | manual |

Modified:
- **OPEN-DIFFUSION-NPU-ONLY:** the host's last step becomes "read the RGBA and encode the
  PNG or JPEG". **Applies to** adds `oflm image`.

Nothing is removed.

## Design

### 1. The model directory (`Atomic-Germ/FLUX.2-klein-4B-NPU2`)

The directory is flat. `remove_model` deletes only top-level files, then fails on a
subdirectory (`model_downloader.cpp:507-523`).

| file | size | notes |
|---|---:|---|
| `config.json` | small | written: `model_type: "flux2-klein"`, the resolutions, the kernel format. `LM_Config::from_pretrained` only needs it to parse (`lm_config.hpp:71-75`) |
| `bundle.json` | 22 KB | `export_bundle.py`'s, with relative paths |
| `schedule_512.json`, `schedule_1024.json` | 113 KB each | |
| `weights.bin` | 7.55 GiB | the 194 packed GEMM weights concatenated, 4 KiB aligned; `bundle.json` gives each one `{offset, bytes}` |
| `embed.bin` | 778 MB | Qwen3-4B's embedding table, bf16 |
| `vae_W.bin`, `vae_S.bin`, `params.bin`, `tf_*.bin`, `dt_*.bin`, `qin_*.bin` | ~130 MB | |
| `tokenizer.json` | 11.4 MB | the checkpoint's |
| `LICENSE`, `README.md` | | Apache 2.0: klein 4B and Qwen3-4B both are |

About 8.9 GB in all.

**One weights file, not 194,** because of how the registry works:
- it lists every file twice (`model_list.json`'s `files` and `model_info.json`);
- `oflm pull` silently skips a listed file that is missing from `model_info.json`
  (`model_downloader.cpp:414-416`).

**The packed weights are tied to the kernels' tile layout** (dit_gemm's bfp16 packing).
`config.json` and the kernel manifest both carry a format hash of that layout, and a
mismatch is refused by name.

### 2. The kernel sets (the family xclbins)

- **What ships:** the runtime subset of the kernel build. That is 6 `final.xclbin` files
  plus instruction streams: 140 files, 14.6 MB, both resolutions. The full build directory
  is 12 GB because of `build/`.
- **Where it goes:** a new `export_dit_kernels.py --install <dir>` copies only those files
  into `src/xclbins/FLUX.2-klein-4B-NPU2/open_kernels/`.
  - It writes `diffusion_kernels.json` there: `format: "oflm-open-diffusion-kernels-v1"`,
    `complete: true`, and the layout hash.
  - The directory is gitignored and shipped by the installer, as every `open_kernels`
    directory is (`.gitignore:84`, `src/inno/oflm.iss:101-102`).
- **How the engine finds it,** the way `find_open_kernels` does for Whisper
  (`whisper_engine_select.cpp:58-91`). It searches in order:
  1. `OFLM_DIFFUSION_KERNELS_DIR`;
  2. `<model_dir>/open_kernels`;
  3. `<root>/xclbins/<bundle family>/open_kernels` for each `utils::xclbin_roots()`.

  A candidate counts only if its manifest is complete and its hash matches the model's.
  Keying on the bundle's `family` lets a fine-tune of the same shape reuse the shipped
  kernels (AGENTS.md).

### 3. `q4nx-build --open-diffusion`

```
q4nx-build --open-diffusion black-forest-labs/FLUX.2-klein-4B -o <dir>
```

One command replaces what is four steps today:
- `generate.py --pack-only`;
- `export_bundle.py`;
- copying `tokenizer.json`;
- writing `config.json`.

Details:
- **Packing needs only the checkpoint and the layout code** (`kp.pack_b_cols`), not a
  kernel build. Today's `<kernels>\packed` is only a cache location.
- **It writes the directory above and `model_info_entry.json`.** It then prints the
  instruction to merge that entry into `src/model_info.json`, as `--open-embedding` does
  (`q4nx/cli.py:199-202`).

### 4. Engine

- **Paths:** everything resolves relative to the model directory. Weights are read from
  `weights.bin` by offset (`engine.cpp:183`).
- **Kernels:** found by the search in section 2.
- **`set_prompt(text)`:** runs the main build's `Tokenizer(model_dir)`
  (`tokenizer.hpp:43-81`) over `bundle.json`'s `prompt_template`, truncates to 512 after
  templating, then calls `set_tokens`.
  - Plain `encode()` is enough: the checkpoint's `post_processor` is ByteLevel and adds no
    special tokens.
  - `Tokenizer` calls `exit(1)` on a missing `tokenizer.json` (`tokenizer.cpp:17-48`), so
    the engine checks for the file first.
- **`encode(format)`:** PNG or JPEG bytes in memory, through `stb_image_write`. It is
  vendored as `third_party/stb/stb_image_write.h` (public domain). Nothing that encodes
  images is vendored today, and vcpkg's FFmpeg on Windows has no zlib. It replaces
  `write_png_rgb`.
- **Device:** the engine keeps its own `xrt::device` (`engine.cpp:87`), because
  `oflm image` runs nothing else. Sharing the app's device belongs to the API PR's
  residency work.
- **`cli.cpp`** stays the standalone gate (`build.cmd`), out of the main build.

### 5. Registry

- **`src/model_list.json`:** add `"flux2-klein": {"4b": {...}}`, modeled on `embed-gemma`:
  - `name`: `FLUX.2-klein-4B-NPU2`, with the Atomic-Germ URLs;
  - the flat `files` list;
  - `"image": true`;
  - `details`: family `flux2-klein`, parameter_size `4B`, quantization_level `bfp16`;
  - `label: ["image-generation"]`;
  - `oflm_min_version`: the release this ships in.
- **`src/model_info.json`:** the merged `model_info_entry.json`.
- **`model_families.hpp:69-70`:** map `flux2-klein` to a non-chat error, so `run` and
  `serve` refuse it before unloading anything (`rest_handler.cpp:466-470`).
- **`model_list.hpp:166-167`, `:196-197`:** skip it in `/api/tags` and `/v1/models`, as
  `whisper-v3` and `embed-gemma` are. The API PR un-hides it from `/v1/models`.
  - These lists skip by literal key, so the NPUE embedding tags (bge, nomic, gte,
    all-minilm) show up in both today. That is a separate issue.

### 6. `oflm image`

```
oflm image <tag> "<prompt>" [-o FILE] [--size 512|1024] [--seed N]
```

- **The prompt** is a third positional (`vm_args.hpp:133-141`), accepted only for
  `image`. Any other command given one is refused, like the bench-embed-only options
  (`:186-194`).
- **`-o/--out`** is new; there is no `-o` today. The default is `oflm-<seed>.png` in the
  current directory.
- **`--size`** defaults to 1024, the API's `auto`.
- **`--seed`** defaults to a random 64-bit value, printed so the image can be reproduced.
- **The tag is checked before any download:** the resolved tag must have `"image": true`.
  `get_model_info`'s llama3.2:1b fallback never applies, as OPEN-WHISPER-SEAM requires for
  `--asrmodel`.
- **Load sequence:** then bench-embed's: resolve, pull if missing, load
  (`benchmark_embed.hpp:296-329`).
- **`main.cpp`:** add `image` to the model-check, pmode and memlock lists (`:522`, `:529`,
  `:553`) and to the dispatch.
- **Output:** the path, the seed, and one timing line like the standalone CLI's, e.g.
  `5.1 s on the NPU (text 0.65, steps 4.0, vae 0.46)`.
- **Build:** under `if(NOT OFLM_USE_HRX)` with `OFLM_USE_OPEN_DIFFUSION=1`, as open_whisper
  is (`CMakeLists.txt:389-451`, `:583`). Only `engine.cpp` joins: the host does no
  compute, so it needs no OpenMP or AVX flags. An HRX build answers "not implemented in
  this build".
- **Docs:** documented in `docs/docs/instructions/cli.md`.

### 7. Tests

- **OPEN-DIFFUSION-TOKENS:**
  - `src/open_diffusion/tokens_test.cpp`, a CMake `add_test` target like
    `benchmark_embed_test`.
  - Golden ids come from `specs/open-diffusion/tests/token_goldens.json`, written by
    `klein_pipeline.token_ids`. It covers the 8 study prompts and one prompt over 512
    tokens.
  - The test reads `tokenizer.json` from the installed model. If the file is absent, the
    test fails naming the path.
- **OPEN-DIFFUSION-DETERMINISM:**
  - `specs/open-diffusion/tests/test_cli_determinism.py`, standard library and
    `subprocess` only.
  - It makes two runs at 512² with seed 1 and checks that the bytes are equal. It needs
    the NPU and the installed model.
- **CLI and PACKAGE:** their manual procedures go in the spec.

### 8. Skill

Update `.opencode/skill/open-diffusion/SKILL.md` with the build, package and pull path, per
AGENTS.md.

## The outward-facing step

- **The upload:** `Atomic-Germ/FLUX.2-klein-4B-NPU2` is about 8.9 GB and public. You
  upload it, or I do once you say so.
- **Until then:** building the directory straight into the models folder tests everything
  except the download. `is_model_downloaded` counts files that are already present.

## Not in this PR

- The step count, one engine holding both resolutions, sharing the device: the API PR.
- `oflm add` for klein derivatives. `oflm-add` requires `model.q4nx`
  (`utilities/oflm-add/oflm_add/__init__.py:49`). It is a follow-up.
- `remove_model` failing on subdirectories, and the literal-key list hiding: separate
  fixes.

## Decisions (user, 2026-09-28)

1. **Command grammar:** `oflm image <tag> "<prompt>"`, with the prompt as a positional.
2. **Default `--size`:** 1024.
3. **The Hugging Face repo: `Cyronius/FLUX.2-klein-4B-NPU2`** (the user's account, not
   `Atomic-Germ`), and `model_list.json` points `oflm pull` at it. The format is interim:
   GGUF support replaces it. Wherever this plan says `Atomic-Germ/FLUX.2-klein-4B-NPU2`,
   read this repo.
4. **int8 GEMM: not a blocker.** It would change `weights.bin` and the layout hash, but
   re-uploading an interim model is cheap. It stays in Phase 7.

## As built (2026-09-28): where it differs from the design above

- **The layout hash** is `export_dit_kernels.layout_hash`: sha256 over every set's stream
  specs (`set_streams`, which `main()` now builds from too) and `WEIGHT_FORMAT`. The
  packer computes it without a kernel build; `--install` hashes the built markers and
  refuses a set built from other specs. Both came out `e0450140c44f78e0`.
- **te_attn's patch words** are read from the kernel set's `fa/dit_fa.json`, not
  `bundle.json`: they belong to the build.
- **Templating and tokenizing** live in `src/open_diffusion/prompt.cpp` (main build only);
  the engine keeps taking ids, so the standalone `build.cmd` needs no tokenizer.
- **`set_tokens` takes the ids' length as given.** It used to stop at the first pad id,
  and a prompt containing `<|endoftext|>` would have been cut short (a goldens case).
- **Sizes are refused before download** from a new registry field, `image_sizes`.
- **`model_info_entry.json` predicts the HF tree API's listing** (git blob sha1 for
  plain files; LFS pointer oid and `lfs.oid` sha256 for `*.bin` and >= 10 MB). The
  whisper builder's plain sha256 `oid` would fail `oflm pull`'s check.
- **VAE weights are packed from the checkpoint** (`vae_decoder.pack_weights`), not from
  `<kernels>/vae_packed.npz`. The model directory was checked byte-identical to the old
  bundle, and the new standalone engine's pixels identical to the old one's.

## Order

1. The engine: relative paths, `weights.bin`, the kernel search, `set_prompt`, `encode`.
2. `export_dit_kernels.py --install` and `q4nx-build --open-diffusion`.
3. The registry entries and the non-chat refusals.
4. CMake and `oflm image`.
5. The tests, docs, skill and spec merge; this plan moves to `archive/`.
6. The Hugging Face upload, then an `oflm image` run on a clean machine.
