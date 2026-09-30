# Plan: the OpenAI Images API over the NPU diffusion engine

Status: **implemented 2026-09-28; merged into `specs/server-api/spec.md` as SERVER-IMAGES-*.**
It is the second of two PRs, on top of the CLI PR (`specs/open-diffusion/archive/cli.md`).

What landed differs from the text below in these ways:
- **The step count** needed a new engine requirement, OPEN-DIFFUSION-STEPS (the step template
  moved on by its derived strides; sigmas from `src/open_diffusion/schedule.hpp`). For the
  bundle's own count the engine uses the bundle's TF/DT bytes, so the default image is
  byte-identical to the CLI PR's.
- **`/v1/models` lists the image model.** OPEN-DIFFUSION-CLI said it did not; that clause moved
  here.
- **Additions:** each `data` item carries its `seed`; `seed: -1` is random (A1111);
  `output_compression` sets the JPEG quality; `response_format: "url"` is refused; an omitted
  `model` is `--imagemodel`'s; `--imagegen 1` allocates both resolutions at startup.
- **Measured (the "to measure first" item):** `--imagegen 1` with llama3.2:1b fits the NPU; with
  `--asr` and `--embed` as well it is not measured.

## The API (the user's decision, 2026-09-27)

- **The primary surface is OpenAI's Images API:**
  - `POST /v1/images/generations` (text to image);
  - `POST /v1/images/edits` (multipart: `image`, optional `mask`).
  - No `/variations`: OpenAI no longer documents it.
- **Honored fields:** `model`, `prompt`, `n`, `size` (`"WxH"`) and `output_format`
  (`png` | `jpeg`; `webp` later, decision 3).
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
| SERVER-IMAGES-GENERATIONS | `POST /v1/images/generations` returns `{created, data: [{b64_json} × n]}`, each a decodable image of `size` in `output_format` (`png` default, `jpeg`); `webp` is a 400 naming it as not implemented | test (integration) |
| SERVER-IMAGES-PARAMS | the alias pairs are one field; both spellings of a pair in one request is a 400 unless the values agree; unknown extras are ignored; types are checked (400 + `param`); `cfg_scale`/`guidance_scale` and `negative_prompt` are accepted and ignored with one log line; `sampler`/`sampler_name` accepts the Euler family and refuses any other name with a 400 that lists the accepted ones | test (unit: a pure `images_request()` in `openai_compat.hpp`, next to `preflight`) |
| SERVER-IMAGES-SIZE | `size` is `WxH` or `auto`; only the engine's resolutions run (512x512, 1024x1024; `auto` = 1024x1024); any other size is a 400 that names the supported ones | test (unit + integration) |
| SERVER-IMAGES-EDITS | `/v1/images/edits` validates its multipart fields, then answers 501 naming what is missing (the NPU VAE encoder), until Phase 8 | test (integration) |
| SERVER-IMAGES-NPU | image requests take the NPU lock like chat; the lock is released on every path; the server keeps serving after an image error | test (integration) |
| SERVER-IMAGES-RESIDENCY | by default an image request swaps the NPU from the chat model to the image engine, and a chat request swaps back; `oflm serve <tag> --imagegen 1 [--imagemodel <tag>]` loads the image engine at startup and keeps it resident beside the chat model; if it cannot be, the server exits at startup naming why | manual |

OPEN-DIFFUSION-DETERMINISM (same prompt, size, steps and seed give the same bytes) is an
engine property. It moves to the CLI plan, whose home spec is `specs/open-diffusion`.

## How each field maps onto klein

- **`model`:** a tag such as `flux2-klein:4b` in `model_list.json`, with `"image": true`
  (added by the CLI PR).
  - Resolution follows SERVER-MODEL-IDENTITY: the tag is resolved before anything is
    unloaded, an unknown tag is a 400 `model_not_found`, and the response names the
    model.
  - The silent llama3.2:1b fallback in `get_model_info` must not apply.
  - Chat-only lists (`/api/tags`) hide it, as they hide `whisper-v3`. `/v1/models` lists
    it.
- **`prompt`:** the engine tokenizes it (the CLI PR), truncating at 512 tokens as diffusers
  does.
- **`n`:** 1-10 images in sequence, seeds `seed + k`. Each is ~5 s (512²) or ~13 s
  (1024²), so `n` > 1 is a long request.
- **`size`:** 512x512 or 1024x1024 (OPEN-DIFFUSION-RESOLUTIONS).
  - Non-square sizes need (W/16)(H/16) to be a multiple of 512 and new stream sets
    (1024x512 qualifies). They are a separate item.
- **`output_format`:** PNG and JPEG through `stb_image_write`, which the CLI PR vendors.
  `webp` is a 400 naming it as not implemented (decision 3).
- **`seed`:** default is a host-random 64-bit seed. The noise is generated on the host,
  which the NPU-only rule allows as setup. The output is deterministic per seed.
- **`steps` / `num_inference_steps`:** klein is distilled for 4 steps, and the default is 4.
  - The streams already allow any count up to 512 (the modulation GEMM's M), so a count
    is just a longer op list and another sigma schedule.
  - The engine instantiates the step template `s` times instead of reading 4 fixed
    steps from the bundle.
  - Allowed range 1-50, otherwise a 400.
- **`cfg_scale` / `guidance_scale`, `negative_prompt`:** accepted and ignored, with one log
  line (decision 1). klein is guidance-distilled: it has no CFG, and diffusers ignores
  guidance for it ("Guidance scale is ignored for step-wise distilled models").
- **`sampler` / `sampler_name`:** the only sampler is flow-match Euler. The Euler family
  maps onto it; any other name is a 400 (decision 2).
- **`partial_images`: not offered.** A preview costs a full VAE decode per step: 0.46 s at
  512², 1.25 s at 1024², about +35% time. That is not cheap.
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

The CLI PR provides:
- the engine in the main build (`OFLM_USE_OPEN_DIFFUSION`);
- tokenizing;
- PNG/JPEG encoding in memory;
- the model directory and `oflm pull`.

This PR adds:

1. **Engine changes:**
   - One `Engine` holds both resolutions. The weights (7.5 GB) are shared, and the
     activations are allocated per resolution (1.4 / 4.6 GiB). Today it is one resolution
     per instance.
   - A step count parameter.
   - No warm-up run at load. The native engine's first image after load is as fast as a
     warm one: 5.12-5.14 s against 5.09-5.15 s at 512², and 13.42 s against 13.36-13.42 s at
     1024².
2. **Routes:**
   - `/v1/images/generations` and `/v1/images/edits` in `create_lm_server()`.
   - Both paths added to `requires_npu_access()`.
   - `send_response` exactly once on every path, or the NPU lock leaks
     (`server.cpp:134-152`).
3. **Residency (decision 4):**
   - **Default: swap.** An image request unloads the chat engine and loads the image
     engine, the way `ensure_model_loaded` switches chat models. A chat request swaps
     back. Load measured 5.2-5.4 s per resolution with the weights in the OS file cache;
     a cold load is not measured.
   - **`--imagegen 1` / `--imagemodel <tag>`** in `vm_args.hpp`, beside `--asr` / `--asrmodel`
     and `--embed` / `--embeddingmodel`. The image engine loads at startup, the way
     `ensure_embed_model_loaded` does, and stays resident. A load failure exits with the
     reason, as `--asr`'s does.
   - Resident still means one request at a time on the NPU (the NPU lock). It saves the
     swap, not the queue.
   - **To measure first:** whether the chat model and the image engine fit the NPU's
     hardware-context and memory limits together, with and without Whisper and the
     embedding model. The image engine alone holds six hardware contexts.
   - The engine opens its own `xrt::device` (`engine.cpp:87`). The other engines share
     the server's device (`npu_device_inst`). Either the engine takes the server's
     device, or two device handles are shown to coexist.
4. **Tests:**
   - `images_request()` unit tests in `openai_compat_test.cpp`: aliases, conflicts, size
     parsing, ranges, the ignored and refused controls.
   - `specs/server-api/tests/test_images_api.py`, standard library only, like its
     neighbours: response shape, decodable image of the right size, size refusal,
     aliases on the wire, edits 501, and the server still serving after an error.

**SERVER-IMAGES-RESIDENCY verification (manual):**
- Without `--imagegen`, send a chat request, then an image request, then a chat request.
  All three succeed.
- With `--imagegen 1`, send the same three. Neither the image engine nor the chat model
  reloads (the server log shows no load line).

## Decisions (the user's, 2026-09-27)

1. **`cfg_scale` and `negative_prompt`:** accept and ignore them, logging one line.
   A1111 clients always send `cfg_scale: 7` and `negative_prompt: ""`.
2. **`sampler`:** accept the Euler family (`euler`, `Euler`, `Euler a`, `flowmatch_euler`)
   as flow-match Euler. Refuse other names with a 400 that lists them.
3. **Output formats:** PNG and JPEG. WebP needs `libwebp` through vcpkg; until a client
   needs it, `webp` is a 400 naming it as not implemented.
4. **Residency:** swap the image engine and the chat model on demand by default.
   `--imagegen 1` keeps both resident.

## Order

1. The CLI PR lands (`specs/open-diffusion/plans/cli.md`).
2. The engine changes.
3. `images_request()` with its unit tests.
4. The generations route, residency, and the integration tests.
5. The edits route, 501 until Phase 8.
6. The spec merge, and this plan moves to archive.
