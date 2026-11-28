---
title: "Whisper"
description: "Local speech-to-text transcription with Whisper Large V3 Turbo on AMD Ryzen AI NPUs."
icon: "graphic_eq"
weight: 60
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "asr" "Speech" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b k="hybrid" t="NPU + CPU" tip="Open Whisper (src/open_whisper/): 32-layer encoder projections run on the NPU; conv/attention and 4-layer token decoder run on the CPU (requires export_whisper_kernels.py from a source checkout until #240 ships)." >}}{{< b k="closed" t="Closed NPU" tip="Default whisper-v3:turbo .q4nx shortcut runs on inherited prebuilt Whisper NPU kernels + libwhisper_npu (omitted on source-built NixOS #244)." >}}{{< b "gguf" "GGUF" >}}{{< b "safetensors" "Safetensors" >}}{{< b "q4nx" "Legacy Q4NX" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "Large V3 Turbo" >}}{{< b k="unsupported" t="Tiny" mono="true" >}}{{< b k="unsupported" t="Base" mono="true" >}}{{< b k="unsupported" t="Small" mono="true" >}}{{< b k="unsupported" t="Medium" mono="true" >}}{{< b k="unsupported" t="Large V3" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "30s" >}}{{< b k="qbest" t="BF16" mono="true" >}}{{< b k="quant" t="F16" mono="true" >}}{{< b k="quant" t="4-bit Fixed" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Windows & Linux release packages: pull the built-in shortcut (runs on Closed NPU; omitted on NixOS #244)
oflm pull whisper-v3:turbo

# Transcribe audio inside interactive chat (/input "path/to/audio.mp3")
oflm run gemma3:4b --asr 1

# Standalone or concurrent OpenAI-compatible transcription server
oflm serve --asr 1
oflm serve gemma3:4b --asr 1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")
with open("audio.mp3", "rb") as f:
    resp = client.audio.transcriptions.create(model="whisper-v3:turbo", file=f)
print(resp.text)
```

From a source checkout with the NPU compiler toolchain installed, compile `Whisper-V3-Turbo-NPU2/open_kernels` first to test the open `NPU + CPU` GGUF engine (`WIP #240`):

```bash
python3 open_kernels/export_whisper_kernels.py
oflm add oxide-lab/whisper-large-v3-turbo-GGUF --family whisper-v3 --xclbin-from Whisper-V3-Turbo-NPU2
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Window & RAM | Status |
|---|---|---|---|---|
| Custom GGUF / Safetensors (`oflm add`) | {{< b "size" "Large V3 Turbo" >}} | {{< b k="hybrid" t="NPU + CPU" tip="32-layer encoder projections run on the NPU; conv/attention and 4-layer token decoder run on the CPU. Requires compiling open_kernels/export_whisper_kernels.py from source until #240 bundles open_kernels." >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} | {{< b "ctx" "30s" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "wip" "240" "WIP #240" >}} |
| `whisper-v3:turbo` | {{< b "size" "Large V3 Turbo" >}} | {{< b k="closed" t="Closed NPU" tip="Downloads model.q4nx and runs on inherited prebuilt Whisper NPU kernels + libwhisper_npu; omitted on source-built NixOS (#244)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "30s" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "240" "GGUF #240" >}} {{< b "wip" "244" "NixOS #244" >}} |

## Tips & Current Limitations

{{< wip issue="240" title="Open Whisper Kernel Packaging & NixOS Support" >}}`utilities/export-kernels.py` does not ship `Whisper-V3-Turbo-NPU2/open_kernels` in prebuilt packages yet (#240), so `oflm pull whisper-v3:turbo` currently runs on the inherited `Closed NPU` runtime (`libwhisper_npu` + git-tracked `src/xclbins`), which is omitted from source-built NixOS packages ([#244](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244)).{{< /wip >}}

- **Supported model & precision:** Only Whisper `Large V3 Turbo` is supported (`Tiny`, `Base`, `Small`, `Medium`, and non-Turbo `Large V3` are not supported). The built-in `whisper-v3:turbo` shortcut uses a 4-bit `.q4nx` archive on `Closed NPU`, while the open Whisper engine (`src/open_whisper/`) reads `llama.cpp`-layout `enc.*`/`dec.*` GGUFs or `model.open.safetensors` in `BF16` or `F16` once `export_whisper_kernels.py` is built.
- **30-second audio window:** Each transcription pass processes a 30-second audio window and emits up to `448` text tokens. Split longer recordings into 30-second segments before calling the API.
- **Open WebUI voice input:** In Open WebUI ([Server Mode](/docs/instructions/server/webui/)), open **Admin Settings -> Audio**, set **Speech-to-Text Engine** to **OpenAI**, URL `http://127.0.0.1:52625/v1`, API key `oflm`, and model `whisper-v3:turbo`.
