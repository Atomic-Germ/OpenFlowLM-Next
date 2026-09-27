# Plan: the OpenAI Images API over the NPU diffusion engine

Status: **for review**. Nothing here is implemented. Written 2026-09-27, after the
first-cut engine landed on branch `feat/open-diffusion`: `src/open_diffusion` produces
FLUX.2 [klein] 4B images fully on the NPU, ~7 s at 512² and ~20 s at 1024².

## The decision (the user's, 2026-09-27)

- **The primary surface is OpenAI's Images API:**
  - `POST /v1/images/generations` (text to image);
  - `POST /v1/images/edits` (multipart: `image`, optional `mask`).
  - No `/variations`: OpenAI no longer documents it.
- **Honored fields:** `model`, `prompt`, `n`, `size` (`"WxH"`) and `output_format`
  (`png` | `jpeg` | `webp`).
- **The response is `{created, data: [{b64_json}]}`.** Base64 only; no `url`.
- **Diffusion controls are optional top-level extras.** Each alias pair is the same field,
  so clients written for vLLM-Omni (diffusers names) and for Lemonade/A1111 both work:
  - `seed`;
  - `steps` = `num_inference_steps`;
  - `cfg_scale` = `guidance_scale`;
  - `negative_prompt`;
  - `sampler` = `sampler_name`.
- **`partial_images` streaming** only if the pipeline can make intermediate images
  cheaply.
- **A1111 shim** (`/sdapi/v1/txt2img`, `/img2img`) only when a specific front-end needs
  it. It would be a thin translation onto the same backend.
- **No ComfyUI `/prompt`.**

## Spec impact

New requirements in `specs/server-api/spec.md` (prefix `SERVER`). They merge there when
implemented:

| ID | what | Verification |
|---|---|---|
| SERVER-IMAGES-GENERATIONS | `POST /v1/images/generations` returns `{created, data: [{b64_json} × n]}`, each a decodable image of `size` in `output_format` | test (integration) |
| SERVER-IMAGES-PARAMS | the alias pairs are one field; both spellings of a pair in one request is a 400 unless the values agree; unknown extras are ignored; types are checked (400 + `param`) | test (unit: a pure `images_request()` in `openai_compat.hpp`, next to `preflight`) |
| SERVER-IMAGES-SIZE | `size` is `WxH` or `auto`; only the engine's resolutions run (512x512, 1024x1024; `auto` = 1024x1024); any other size is a 400 that names the supported ones | test (unit + integration) |
| SERVER-IMAGES-EDITS | `/v1/images/edits` validates its multipart fields, then answers 501 naming what is missing (the NPU VAE encoder), until Phase 8 | test (integration) |
| SERVER-IMAGES-NPU | image requests take the NPU lock like chat; the lock is released on every path; the server keeps serving after an image error | test (integration) |
| OPEN-DIFFUSION-DETERMINISM | the same prompt, size, steps and seed give the same bytes (measured: yes, run to run and Python vs native) | test (integration, 512², one image) |

## How each field maps onto klein

- **`model`:** a tag such as `flux2-klein:4b` in `model_list.json`, with `"image": true`.
  - Resolution follows SERVER-MODEL-IDENTITY: the tag is resolved before anything is
    unloaded, an unknown tag is a 400 `model_not_found`, and the response names the
    model.
  - The silent llama3.2:1b fallback in `get_model_info` must not apply.
  - Chat-only lists (`/api/tags`) hide it, as they hide `whisper-v3`. `/v1/models` lists
    it.
- **`prompt`:** tokenized with the main build's `Tokenizer(bundle_dir)` over the bundle's
  `prompt_template`. Qwen3's chat template must be byte-exact, and the pipeline's ids are
  already verified against transformers.
  - `Tokenizer` calls `exit(1)` on a missing tokenizer.json, so check the file first.
  - Truncate at 512 tokens, as diffusers does.
- **`n`:** 1-10 images in sequence, seeds `seed + k`. Each is ~7 s (512²) or ~20 s
  (1024²), so `n` > 1 is a long request.
- **`size`:** 512x512 or 1024x1024 (OPEN-DIFFUSION-RESOLUTIONS).
  - Non-square sizes need (W/16)(H/16) to be a multiple of 512 and new stream sets
    (1024x512 qualifies). They are a separate item.
- **`output_format`:**
  - PNG and JPEG through `stb_image_write` (public domain, one vendored header, no DLL).
    The engine's stored-block PNG is 3 MB at 1024² and too large for base64.
  - WebP needs `libwebp`, a new vcpkg dependency (decision 3). Until then, `webp` is a
    400 naming it as not implemented.
- **`seed`:** default is a host-random 64-bit seed. The noise is generated on the host,
  which the NPU-only rule allows as setup. The output is deterministic per seed.
- **`steps` / `num_inference_steps`:** klein is distilled for 4 steps, and the default is 4.
  - The streams already allow any count up to 512 (the modulation GEMM's M), so a count
    is just a longer op list and another sigma schedule.
  - The engine instantiates the step template `s` times instead of reading 4 fixed
    steps from the bundle.
  - Allowed range 1-50, otherwise a 400.
- **`cfg_scale` / `guidance_scale`, `negative_prompt`:** klein is guidance-distilled. It has
  no CFG, and diffusers ignores guidance for it ("Guidance scale is ignored for step-wise
  distilled models"). See decision 1.
- **`sampler` / `sampler_name`:** the only sampler is flow-match Euler. See decision 2.
- **`partial_images`: not offered.** A preview costs a full VAE decode per step: 0.6 s at
  512², 1.9 s at 1024², +40% time. That is not cheap.
  - A request with `stream: true` or `partial_images` > 0 is a 400 naming it.
  - It is revisited if TAEF2 (the tiny FLUX.2 decoder, which runs on `dit_conv`) lands as
    a preview decoder.
- **`/v1/images/edits`:** klein edits by appending the input image's VAE latents as
  reference tokens. That needs a VAE *encoder* on the NPU, and a DiT stream set per
  (size, reference size), since the joint sequence grows by the reference tokens. `mask`
  is the inpaint pipeline on top.
  - Phase 8. Until then: parse and validate, then answer 501.
  - `parse_multipart` needs three fixes for this endpoint:
    - quoted boundaries;
    - filling `content_type`;
    - repeated `image[]` parts, which overwrite each other today.

## Server work

1. **Build:** add `open_diffusion/engine.cpp` to `src/CMakeLists.txt`, the way
   open_whisper enters, with an `OFLM_USE_OPEN_DIFFUSION` define. `cli.cpp` stays out.
2. **Engine changes:**
   - One `Engine` holds every resolution, with the weights (7.5 GB) shared and the
     activations allocated per resolution (1.4 / 4.6 GiB). Today it is one resolution per
     instance.
   - A step count parameter.
   - A warm-up run at load: the first image is ~1.5× slower.
   - `rgb()` into memory.
3. **Routes:**
   - `/v1/images/generations` and `/v1/images/edits` in `create_lm_server()`.
   - Both paths added to `requires_npu_access()`.
   - `send_response` exactly once on every path, or the NPU lock leaks
     (`server.cpp:134-152`).
4. **Residency:** the engine is ~12 GB of buffers and six hardware contexts. Nothing
   manages residency between model kinds today.
   - Proposed: load on the first image request after unloading the chat engine, the way
     `ensure_model_loaded` switches chat models.
   - A chat request unloads it again.
   - Load is ~20 s, dominated by reading 7.5 GB of weights.
5. **Packaging:** the server reads a model directory, not `C:\dev`.
   - `q4nx-build --open-diffusion` writes the bundle into it, with a `config.json`, which
     `is_model_downloaded` requires.
   - `oflm add` installs it with the family xclbins under
     `src/xclbins/FLUX.2-klein-4B-NPU2/open_kernels/`.
   - This comes first: nothing can be served until a model directory exists.
6. **Tests:**
   - `images_request()` unit tests in `openai_compat_test.cpp`: aliases, conflicts, size
     parsing, ranges.
   - `specs/server-api/tests/test_images_api.py`, standard library only, like its
     neighbours: response shape, decodable image of the right size, size refusal,
     aliases on the wire, edits 501, and the server still serving after an error.

## Decisions for you

1. **`cfg_scale` and `negative_prompt` on a distilled model.**
   - Recommended: accept and ignore them, logging one line, as diffusers does. A1111
     clients always send `cfg_scale: 7` and `negative_prompt: ""`, so refusing would
     break the clients this alias set exists for.
   - Alternative: refuse any non-default value.
2. **An unsupported `sampler`.**
   - Recommended: accept the Euler family (`euler`, `Euler`, `Euler a`, `flowmatch_euler`)
     as flow-match Euler, and refuse other names with a 400 that lists them. No silent
     substitution: a client asking for DPM++ would otherwise get Euler without knowing.
3. **WebP.** It needs `libwebp` via vcpkg. Approve the dependency, or ship PNG/JPEG first
   with `webp` refused by name (recommended until a client needs it).
4. **Residency.** Swap the diffusion engine and the chat model on demand (recommended), or
   keep both resident. The latter is untested against the NPU's context and memory
   limits.

## Order

1. Packaging (a model directory).
2. The engine changes.
3. `images_request()` with its unit tests.
4. The generations route and the integration tests.
5. The edits route, 501 until Phase 8.
6. The spec merge, and this plan moves to archive.
