---
title: "Supported Models"
description: "Browse supported model families for Hugging Face GGUF downloads via oflm add and built-in shortcuts via oflm pull."
url: "/models/"
kicker: "Model Catalog"
icon: "view_module"
ctas:
  - text: "Model Guides"
    url: "/docs/models/"
    icon: "menu_book"
  - text: "NPU Runtimes"
    url: "/engines/"
    icon: "memory"
    outline: true
---

{{< models-explorer >}}

## Installing Custom Checkpoints

`oflm add` inspects the GGUF header and matches Hugging Face repositories, community fine-tunes, and local `.gguf` files to compiled open NPU kernel sets. Starred quantizations ({{< b k="qbest" t="Q4_1" mono="true" >}} for chat and vision, {{< b k="qbest" t="Q8_0" mono="true" >}} for embeddings) load without conversion for the fastest startup; other listed GGUF quantizations (such as `Q4_K_M`) convert on first load.

```bash
# Install Hugging Face GGUF repositories
oflm add unsloth/Qwen3.5-4B-GGUF
oflm add unsloth/gemma-3-4b-it-GGUF
oflm add ibm-granite/granite-4.0-micro-GGUF

# Register a local model folder without copying weights
oflm add ./my-local-model --family gemma3 --xclbin-from Gemma3-4B-NPU2
```

## Legacy Q4NX Shortcuts & Closed NPU Phase-Out

<div class="oflm-pill-row mb-2">
<span class="oflm-group-label">Status:</span>
{{< b k="closed" t="Closed NPU" tip="Unported .q4nx shortcuts run on inherited prebuilt NPU kernels and closed runtime libraries (omitted on source-built NixOS #244)." >}}
{{< b k="wip" issue="240" t="Target Shortcuts to GGUF #240" >}}
{{< b k="wip" issue="236" t="Gemma 4 E2B/E4B #236" >}}
{{< b k="wip" issue="91" t="GPT-OSS 20B #91" >}}
{{< b k="wip" issue="73" t="Gemma & Omni Multimodal #73" >}}
{{< b k="wip" issue="244" t="NixOS Prebuilts #244" >}}
{{< b k="q4nx" t="Legacy Q4NX" >}}
</div>

Earlier OpenFlowLM releases packaged shortcuts in a `.q4nx` weight container (`Atomic-Germ/*-NPU2`). Most `.q4nx` shortcuts (`qwen3*`, `qwen3.5*`, `qwen3.6-moe*`, `llama3*`, `lfm2*`, `phi4-mini-it:4b`, `deepseek-r1*`, and text on `gemma3:4b` and `gemma4-it:12b`) now run on **Open `NPU`** or **`NPU + CPU`** kernels. A small remaining set (`whisper-v3:turbo`, `gemma4-it:e2b/e4b`, `gpt-oss:20b`, `qwen2.5vl-it:3b`, and Gemma/MedGemma/TranslateGemma `.q4nx` vision) still uses inherited **`Closed NPU`** prebuilt binaries and closed runtime libraries: those shortcuts cannot load GGUFs, require an explicit `OFLM_<FAMILY>_ENGINE=closed` opt-in where open engines exist (no silent fallback), and are omitted from source-built NixOS packages (`#244`). For any new model or fine-tune, pass a compatible GGUF repository to `oflm add`.
