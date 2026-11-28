---
title: "FLUX.2"
description: "Local image generation and editing with FLUX.2 [klein] 4B on AMD Ryzen AI NPUs."
icon: "brush"
weight: 50
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "image" "Image" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "open" "NPU" >}}{{< b "bundle" "Pre-packed" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "klein 4B" >}}{{< b k="unsupported" t="klein 9B" mono="true" >}}{{< b k="unsupported" t="dev 32B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "512x512" >}}{{< b "ctx" "1024x1024" >}}{{< b k="qbest" t="BF16 + 8-bit" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Download the pre-packed NPU bundle (~12.7 GB)
oflm pull flux2-klein:4b

# Generate a 1024x1024 image (pass --size 512 for 512x512 output)
oflm image flux2-klein:4b "a red fox in fresh snow" --seed 1 -o fox.png

# Edit an existing image with a text instruction (--image)
oflm image flux2-klein:4b "turn the street into a rainy neon night scene" \
  --image street.jpg --size 512 --seed 42 -o street_night.png
```

Start `oflm serve` to expose OpenAI-compatible `POST /v1/images/generations` and `POST /v1/images/edits` endpoints:

```bash
oflm serve
```

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")
resp = client.images.generate(
    model="flux2-klein:4b",
    prompt="a ceramic teacup on a sunlit wooden table",
    size="1024x1024",
)
with open("teacup.png", "wb") as f:
    f.write(base64.b64decode(resp.data[0].b64_json))
```

See the [Image Generation & Editing Guide](/docs/instructions/server/images/) for `--steps`, `--seed`, and `output_format` options.

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Resolution & RAM | Status |
|---|---|---|---|---|
| Custom GGUF (`oflm add`) | {{< b "size" "klein 4B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "512x512" >}} {{< b "ctx" "1024x1024" >}} {{< b "size" "32 GB+ RAM" >}} | {{< b "wip" "240" "Direct GGUF #240" >}} |
| `flux2-klein:4b` | {{< b "size" "klein 4B" >}} | {{< b "open" "NPU" >}} {{< b "bundle" "Pre-packed" >}} | {{< b "ctx" "512x512" >}} {{< b "ctx" "1024x1024" >}} {{< b "size" "32 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |

## Tips & Current Limitations

{{< wip issue="240" title="Direct Community GGUF Loading" >}}Issue #240 tracks loading community GGUF releases (such as `city96/FLUX.2-klein-4B-gguf`) via `oflm add` without a source-tree packing step.{{< /wip >}}

- **Pre-packed precision:** The `flux2-klein:4b` bundle ships pre-packed in mixed `BF16` and 8-bit precision and runs with no local conversion step.
- **Supported resolutions:** OpenFlowLM supports `512x512` and `1024x1024` for generation and editing, and rejects other dimensions such as `768x768`. When you pass `--image`, OpenFlowLM resizes and center-crops the reference image to match the target size.
- **Model size & RAM:** Only `klein 4B` is supported (`klein 9B` and `dev 32B` are not supported). Plan on 32 GB+ of system RAM for the ~12.7 GB text encoder, diffusion transformer, and VAE bundle.
