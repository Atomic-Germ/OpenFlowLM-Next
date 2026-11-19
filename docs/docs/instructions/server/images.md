---
title: "Image Generation and Editing"
description: "Generate and edit images locally on your Ryzen AI NPU with FLUX.2 klein 4B via the CLI and Images API."
icon: "image"
weight: 25
aliases: ["/docs/instructions/images/"]
---

Generate and edit `512x512` and `1024x1024` images locally on your Ryzen AI XDNA2 NPU with FLUX.2 [klein] 4B (`flux2-klein:4b`) via `oflm image` and `/v1/images/*` (~3.7 s at `512x512` and ~12.0 s at `1024x1024` in 4 steps with `turbo` power mode).

## Download the Model

Download `flux2-klein:4b` ahead of time with `oflm pull` (or let `oflm image` download it on first use). Use `32 GB+` of system RAM when generating `1024x1024` images or keeping a chat model resident alongside diffusion:

```bash
oflm pull flux2-klein:4b
```

{{< wip issue="240" title="Diffusion GGUF Packaging" >}}`flux2-klein:4b` currently downloads as a verified ~9.1 GB NPU bundle (`BF16 + 8-bit`) via `oflm pull flux2-klein:4b`. Direct single-file GGUF loading is tracked in #240.{{< /wip >}}

## CLI Usage (`oflm image`)

Generate new images from a text prompt or pass `--image` to edit an existing `.png` or `.jpg` photo:

```bash
# Text-to-image (1024x1024 PNG or 512x512 JPEG)
oflm image flux2-klein:4b "a red fox in fresh snow" -o fox.png
oflm image flux2-klein:4b "a lighthouse at dusk" --size 512 --seed 7 -o lighthouse.jpg

# Reference-image editing
oflm image flux2-klein:4b "Make it a watercolor painting" --image coffee.jpg --seed 7 -o coffee_watercolor.png
```

| Flag | Default | Description |
|---|---|---|
| `-o`, `--out FILE` | `oflm-<seed>.png` | Output `.png`, `.jpg`, or `.jpeg` path |
| `--size PIXELS` | `1024` (auto on edits) | `512` or `1024` square pixels |
| `--seed N` | random 64-bit int | Reproducible output for identical prompt, size, steps, and seed |
| `--steps N` | `0` (uses `4`) | Denoising steps (`1` to `50`; `4` is recommended for FLUX.2 [klein] 4B) |
| `--image FILE` | *(none)* | Reference `.png` or `.jpg` photo to edit (auto-oriented via EXIF and center-cropped) |
| `--modelscope` | `false` | Download model files from ModelScope if missing |

Invalid reference images (unsupported formats, dimensions under 64 px, aspect ratios wider than 8:1, or resolutions above 64 MP) are rejected immediately before loading weights.

> [!NOTE]
> `flux2-klein:4b` is an image diffusion model rather than a text chat model (`oflm run` rejects it). To serve `POST /v1/images/generations` and `POST /v1/images/edits` over HTTP, start `oflm serve --imagegen 1` (dedicated image server) or `oflm serve <chat-tag> --imagegen 1` (co-resident chat + image server).

## OpenAI Images API (`oflm serve`)

Both `POST /v1/images/generations` and `POST /v1/images/edits` run on `oflm serve`. Pass `--imagegen 1` to preload `flux2-klein:4b` at startup (alone or alongside a chat model; without `--imagegen 1`, OpenFlowLM swaps any loaded chat model out during an image request and reloads it afterward):

```bash
# Dedicated image server (generations + edits):
oflm serve --imagegen 1

# Co-resident chat + image server:
oflm serve llama3.2:1b --imagegen 1
```

`GET /v1/models` includes `flux2-klein:4b` for OpenAI SDKs, while `GET /api/tags` omits it so Ollama chat interfaces do not list it in chat model pickers.

<div class="oflm-two-col">
  <div class="oflm-card">

### Generate (`POST /v1/images/generations`)

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")
res = client.images.generate(
    model="flux2-klein:4b",
    prompt="a red fox in fresh snow",
    size="1024x1024",
    output_format="png",
    extra_body={"seed": 1, "steps": 4},
)
with open("fox.png", "wb") as f:
    f.write(base64.b64decode(res.data[0].b64_json))
```

  </div>
  <div class="oflm-card">

### Edit (`POST /v1/images/edits`)

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")
with open("coffee.jpg", "rb") as img:
    res = client.images.edit(
        model="flux2-klein:4b",
        image=img,
        prompt="Make it a watercolor painting",
        size="512x512",
        extra_body={"seed": 7},
    )
with open("watercolor.png", "wb") as f:
    f.write(base64.b64decode(res.data[0].b64_json))
```

  </div>
</div>

Equivalent `curl` requests:

```bash
# Text-to-image generation
curl -s http://127.0.0.1:52625/v1/images/generations -H "Content-Type: application/json" \
  -d '{"model":"flux2-klein:4b","prompt":"a lighthouse at dusk","size":"512x512","seed":7}' \
  | jq -r '.data[0].b64_json' | base64 -d > lighthouse.png

# Reference-image edit (multipart upload up to 32 MB PNG/JPEG)
curl -s http://127.0.0.1:52625/v1/images/edits \
  -F "model=flux2-klein:4b" -F "image=@coffee.jpg" \
  -F "prompt=Make it a watercolor painting" -F "size=512x512" -F "seed=7" \
  | jq -r '.data[0].b64_json' | base64 -d > coffee_watercolor.png
```

### API Parameters & Tips

| Parameter | Accepted Values | Behavior |
|---|---|---|
| `model` | `"flux2-klein:4b"` (default `--imagemodel`) | Unknown or chat model tags return HTTP 400 `model_not_found` |
| `size` | `"512x512"`, `"1024x1024"`, `"auto"` | Non-square or unsupported sizes return HTTP 400 |
| `n` | `1` to `10` (default `1`) | Generates `n` images using sequential seeds (`seed, seed+1, ...`), returned in `data[i].seed` |
| `output_format` / `output_compression` | `"png"` (default), `"jpeg"` (`1` to `100`, default `90`) | `"webp"` returns HTTP 400 `not_implemented`; `response_format` must be `"b64_json"` |
| `steps` / `num_inference_steps` | `1` to `50` (default `4`) | Must match if both fields are sent |
| `sampler` / `sampler_name` | Euler variants (`"euler"`, `"flowmatch_euler"`, etc.) | Other samplers return HTTP 400; `cfg_scale` and `negative_prompt` are ignored |

- **Editing scope:** Edits take a single reference image; `mask` inpainting and multiple `image[]` uploads return HTTP 400 `not_implemented`.
- **Performance tip:** Pass `--pmode turbo` (or set `oflm config set --config-key pmode --config-value turbo`) for the fastest generation times.
