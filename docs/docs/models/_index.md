---
title: "Models"
description: "Supported model families, GGUF installation with oflm add, and built-in download shortcuts."
icon: "view_module"
weight: 30
---

<div class="oflm-two-col">
  <div class="oflm-card">
    <h3>1. Bring Any Compatible Model (<code>oflm add</code>)</h3>
    <p class="mb-2">Run any compatible Hugging Face checkpoint, community fine-tune, or merged model with <code>oflm add &lt;hf-repo-or-file&gt;</code>. OpenFlowLM inspects the model header and pairs it with compiled open AMD Ryzen AI NPU kernels.</p>
    <div class="small fw-semibold mb-1" style="color: var(--oflm-muted);">Supported Formats</div>
    <div class="oflm-pill-row mb-2">{{< b "gguf" "GGUF" >}}{{< b "safetensors" "Safetensors" >}}{{< b "bundle" "Pre-packed" >}}</div>
    <div class="small fw-semibold mb-1" style="color: var(--oflm-muted);">Chat &amp; Vision Quantization</div>
    <div class="mb-2">{{< quants "llm" >}}</div>
    <div class="small fw-semibold mb-1" style="color: var(--oflm-muted);">Embedding Quantization</div>
    <div>{{< quants "embed" >}}</div>
  </div>
  <div class="oflm-card">
    <h3>2. Built-in Download Shortcuts (<code>oflm pull</code>)</h3>
    <p class="mb-2">The installer bundles no model weights. Short tags download pre-configured Hugging Face checkpoints on demand with <code>oflm pull &lt;tag&gt;</code> or on your first <code>oflm run &lt;tag&gt;</code>.</p>
    <div class="small fw-semibold mb-1" style="color: var(--oflm-muted);">Shortcut Formats</div>
    <div class="oflm-pill-row mb-2">{{< b "gguf" "GGUF" >}}{{< b "safetensors" "Safetensors" >}}{{< b "bundle" "Pre-packed" >}}{{< b "q4nx" "Legacy Q4NX" >}}</div>
    <div class="small fw-semibold mb-1" style="color: var(--oflm-muted);">Shortcut Roadmap</div>
    <div class="oflm-pill-row mb-2">{{< b "wip" "240" "Target Shortcuts to GGUF #240" >}}{{< b "wip" "244" "NixOS Prebuilts #244" >}}</div>
    <p class="small mb-0" style="color: var(--oflm-muted);">Most <code>.q4nx</code> shortcuts run on open NPU kernels; remaining unported <code>.q4nx</code> shortcuts use temporary <code>Closed NPU</code> prebuilts (omitted on NixOS <code>#244</code>) while open kernels land.</p>
  </div>
</div>

{{< wip issue="240" title="GGUF Registry Updates & Open Kernel Roadmap" >}}Issue #240 tracks GGUF registry updates and production validation across target families. Related issues track open text kernels for Gemma 4 `E2B` and `E4B` ([#236](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/236)), open vision and audio for Gemma 4 and Qwen3.5-Omni ([#73](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/73)), open NPU kernels for GPT-OSS 20B ([#91](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/91)), and prebuilt `src/xclbins` exclusion on source-built NixOS ([#244](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244)).{{< /wip >}}

## Model Families

| Family | Capabilities | Variants | Runtime & Format | Status |
|---|---|---|---|---|
| [Qwen](/docs/models/qwen/) | {{< b "llm" "Chat" >}} {{< b "vlm" "Vision" >}} {{< b "think" "Reasoning" >}} {{< b "tools" "Tools" >}} | {{< b "size" "0.6B" >}} {{< b "size" "0.8B" >}} {{< b "size" "1.7B" >}} {{< b "size" "2B" >}} {{< b "size" "3B (Qwen 2.5)" >}} {{< b "size" "4B" >}} {{< b "size" "VL 4B" >}} {{< b "size" "8B" >}} {{< b "size" "VL 8B" >}} {{< b "size" "9B" >}} {{< b "size" "14B" >}} {{< b "size" "27B" >}} {{< b "size" "35B-A3B" >}} {{< b k="unsupported" t="0.5B" mono="true" >}} {{< b k="unsupported" t="1.5B" mono="true" >}} {{< b k="unsupported" t="7B" mono="true" >}} {{< b k="unsupported" t="14B (Qwen 2.5)" mono="true" >}} {{< b k="unsupported" t="30B-A3B" mono="true" >}} {{< b k="unsupported" t="32B" mono="true" >}} {{< b k="unsupported" t="72B" mono="true" >}} {{< b k="unsupported" t="122B-A10B" mono="true" >}} {{< b k="unsupported" t="235B-A22B" mono="true" >}} {{< b k="unsupported" t="397B-A17B" mono="true" >}} | {{< b "open" "NPU" >}} {{< b k="hybrid" t="NPU + CPU" tip="Qwen 3.5 and Qwen 3.6 MoE run attention and FFN/expert projections on the NPU; MoE routing, multi-token prompt recurrent state updates, and ViT image encoding (across Qwen-VL, Qwen 3.5, and Qwen 3.6 MoE) run on the CPU." >}} {{< b k="closed" t="Closed NPU" tip="Only the legacy qwen2.5vl-it:3b .q4nx shortcut requires OFLM_QWEN_ENGINE=closed (omitted on NixOS #244); Qwen 2.5-VL 3B GGUFs run on the open runtime." >}} {{< b "gguf" "GGUF" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "240" "Shortcuts #240" >}} |
| [Gemma](/docs/models/gemma/) | {{< b "llm" "Chat" >}} {{< b "vlm" "Vision" >}} {{< b "asr" "Speech" >}} {{< b "think" "Reasoning" >}} {{< b "tools" "Tools" >}} | {{< b "size" "4B" >}} {{< b "size" "12B" >}} {{< b k="wip" issue="236" t="E2B #236" >}} {{< b k="wip" issue="236" t="E4B #236" >}} {{< b k="unsupported" t="270M" mono="true" >}} {{< b k="unsupported" t="1B" mono="true" >}} {{< b k="unsupported" t="26B-A4B" mono="true" >}} {{< b k="unsupported" t="27B" mono="true" >}} {{< b k="unsupported" t="31B" mono="true" >}} | {{< b "open" "NPU" >}} {{< b k="closed" t="Closed NPU" tip="Gemma 4 E2B/E4B .q4nx shortcuts and vision/audio on gemma4-it:12b or gemma3:4b use inherited prebuilt kernels (OFLM_GEMMA4_ENGINE=closed / OFLM_GEMMA_ENGINE=closed; omitted on NixOS #244)." >}} {{< b "gguf" "GGUF" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "236" "E2B/E4B #236" >}} {{< b "wip" "73" "Multimodal #73" >}} |
| [IBM Granite](/docs/models/granite/) | {{< b "llm" "Chat" >}} {{< b "think" "Reasoning" >}} {{< b "tools" "Tools" >}} | {{< b "size" "3B" >}} {{< b k="unsupported" t="350M" mono="true" >}} {{< b k="unsupported" t="1B" mono="true" >}} {{< b k="unsupported" t="4.0-H Hybrid" mono="true" >}} {{< b k="unsupported" t="7B-A1B" mono="true" >}} {{< b k="unsupported" t="8B" mono="true" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "240" "Shortcut #240" >}} {{< b "wip" "189" "Drafter #189" >}} |
| [LiquidAI LFM](/docs/models/lfm/) | {{< b "llm" "Chat" >}} {{< b "think" "Reasoning" >}} {{< b "tools" "Tools" >}} | {{< b "size" "1.2B" >}} {{< b "size" "2.6B" >}} {{< b k="unsupported" t="350M" mono="true" >}} {{< b k="unsupported" t="700M" mono="true" >}} {{< b k="unsupported" t="8B-A1B" mono="true" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Short-convolution projections, gated attention, and FFN layers run on the NPU; short-conv state carry during prompt prefill runs on the CPU." >}} {{< b "gguf" "GGUF" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ok" "Ready" >}} |
| [FLUX.2](/docs/models/flux/) | {{< b "image" "Image" >}} | {{< b "size" "klein 4B" >}} {{< b k="unsupported" t="klein 9B" mono="true" >}} {{< b k="unsupported" t="dev 32B" mono="true" >}} | {{< b "open" "NPU" >}} {{< b "bundle" "Pre-packed" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "240" "GGUF #240" >}} |
| [Whisper](/docs/models/whisper/) | {{< b "asr" "Speech" >}} | {{< b "size" "Large V3 Turbo" >}} {{< b k="unsupported" t="Tiny" mono="true" >}} {{< b k="unsupported" t="Base" mono="true" >}} {{< b k="unsupported" t="Small" mono="true" >}} {{< b k="unsupported" t="Medium" mono="true" >}} {{< b k="unsupported" t="Large V3" mono="true" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Open Whisper (src/open_whisper/): 32-layer encoder projections run on the NPU; conv/attention and 4-layer token decoder run on the CPU (requires export_whisper_kernels.py from source until #240 ships)." >}} {{< b k="closed" t="Closed NPU" tip="Default whisper-v3:turbo .q4nx shortcut runs on inherited prebuilt Whisper NPU kernels and closed library (omitted on source-built NixOS #244)." >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "240" "Open Kernels #240" >}} {{< b "wip" "244" "NixOS #244" >}} |
| [BERT, BGE, Nomic & GTE](/docs/models/bert/) | {{< b "embed" "Embeddings" >}} | {{< b "size" "22M" >}} {{< b "size" "33M" >}} {{< b "size" "109M" >}} {{< b "size" "137M" >}} {{< b "size" "305M" >}} {{< b "size" "335M" >}} {{< b k="unsupported" t="BGE-M3 (568M)" mono="true" >}} {{< b k="unsupported" t="Qwen2-Embed (1.5B/7B)" mono="true" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} | {{< b "ok" "Ready" >}} |
| [EmbeddingGemma](/docs/models/embeddinggemma/) | {{< b "embed" "Embeddings" >}} | {{< b "size" "300M" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Linear and FFN projections run on the NPU (npu_matmul_f32); bidirectional attention and vector pooling across 2,048-token inputs run on the CPU (falls back to CPU on source-built NixOS #244)." >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} | {{< b "ok" "Ready" >}} |
| [Llama 3](/docs/models/llama/) | {{< b "llm" "Chat" >}} | {{< b "size" "1B" >}} {{< b "size" "3B" >}} {{< b "size" "8B" >}} {{< b k="unsupported" t="11B-Vision" mono="true" >}} {{< b k="unsupported" t="70B" mono="true" >}} {{< b k="unsupported" t="90B" mono="true" >}} {{< b k="unsupported" t="405B" mono="true" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ok" "Ready" >}} |
| [Phi](/docs/models/phi/) | {{< b "llm" "Chat" >}} | {{< b "size" "4B" >}} {{< b k="unsupported" t="3.8B" mono="true" >}} {{< b k="unsupported" t="5.6B Multimodal" mono="true" >}} {{< b k="unsupported" t="14B" mono="true" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ok" "Ready" >}} |
| [MedGemma](/docs/models/medgemma/) | {{< b "llm" "Chat" >}} {{< b "vlm" "Vision" >}} | {{< b "size" "4B" >}} {{< b k="unsupported" t="27B" mono="true" >}} | {{< b "open" "NPU" >}} {{< b k="closed" t="Closed NPU" tip="Built-in medgemma:4b and medgemma1.5:4b .q4nx shortcuts require OFLM_GEMMA_ENGINE=closed (lacks open_kernels symlink and uses closed gemma_npu for vision; omitted on NixOS #244)." >}} {{< b "gguf" "GGUF" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "73" "Vision #73" >}} |
| [TranslateGemma](/docs/models/translategemma/) | {{< b "llm" "Chat" >}} {{< b "vlm" "Vision" >}} | {{< b "size" "4B" >}} {{< b "size" "12B" >}} {{< b k="unsupported" t="27B" mono="true" >}} | {{< b "open" "NPU" >}} {{< b k="closed" t="Closed NPU" tip="Built-in translategemma:4b .q4nx shortcut requires OFLM_GEMMA_ENGINE=closed (lacks open_kernels symlink and uses closed gemma_npu for vision; omitted on NixOS #244)." >}} {{< b "gguf" "GGUF" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "73" "Vision #73" >}} |
| [DeepSeek](/docs/models/deepseek/) | {{< b "llm" "Chat" >}} {{< b "think" "Reasoning" >}} | {{< b "size" "8B" >}} {{< b k="unsupported" t="1.5B" mono="true" >}} {{< b k="unsupported" t="7B" mono="true" >}} {{< b k="unsupported" t="14B" mono="true" >}} {{< b k="unsupported" t="32B" mono="true" >}} {{< b k="unsupported" t="70B" mono="true" >}} {{< b k="unsupported" t="671B" mono="true" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ok" "Ready" >}} |
| [Nanbeige](/docs/models/nanbeige/) | {{< b "llm" "Chat" >}} {{< b "think" "Reasoning" >}} {{< b "tools" "Tools" >}} | {{< b "size" "3B" >}} {{< b k="unsupported" t="32B" mono="true" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ok" "Ready" >}} |
| [GPT-OSS](/docs/models/gpt-oss/) | {{< b "llm" "Chat" >}} {{< b "think" "Reasoning" >}} | {{< b "size" "20B" >}} {{< b k="wip" issue="91" t="20B Open GGUF #91" >}} {{< b k="unsupported" t="120B" mono="true" >}} | {{< b k="closed" t="Closed NPU" tip="Requires OFLM_GPTOSS_ENGINE=closed to run inherited prebuilt NPU kernels on .q4nx shortcuts (omitted on source-built NixOS #244; >32 GB RAM recommended)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "wip" "91" "Open NPU #91" >}} {{< b "wip" "240" "GGUF #240" >}} |

**Additional Shipped Open Dense Geometries (`oflm add`):** Beyond the 15 families above, OpenFlowLM compiles open dense NPU kernel sets in `open_kernels/recipes/specs/` for **Hunyuan-MT (`7B`)**, **K2 / Horizon (`3.7B`, `7B`)**, and **MiniCPM (`2B`)**: pass any compatible GGUF checkpoint to `oflm add`.

## Installing Custom Hugging Face Checkpoints

`oflm add` reads the GGUF header and matches your checkpoint against supported model geometries. Fine-tunes, abliterated checkpoints, and layer-scaled merges (such as a 40B merge built on Qwen 3.5 27B layer dimensions) run without waiting for a release.

Pass `--family` and `--xclbin-from` when installing from a local folder or a custom repository whose name hides the base architecture:

```bash
# Install a Hugging Face GGUF checkpoint
oflm add unsloth/Qwen3-4B-GGUF
oflm run qwen3:4b

# Layer-scaled 40B merge matches the 27B kernel set
oflm add mradermacher/Qwen3.8-40B-Ultimate-Heretic-Uncensored-Build2-i1-GGUF

# Explicit family and kernel matching for local folders or custom names
oflm add <repo-or-dir> --family qwen3.6-moe --xclbin-from Qwen3.6-35B-A3B-NPU2
```

## Serving Embeddings Alongside Chat

Pass `--embed 1` to `oflm serve` to load [BERT, BGE, Nomic & GTE](/docs/models/bert/) (512-token batched indexing) or [EmbeddingGemma](/docs/models/embeddinggemma/) (2,048-token sequences) next to any chat model in [Server Mode](/docs/instructions/server/):

```bash
oflm serve llama3.2:1b --embed 1 --embeddingmodel bge-base:en-v1.5
oflm serve llama3.2:1b --embed 1 --embeddingmodel nomic-embed-text:v1.5 --prompt-name query
oflm serve gemma3:4b --embed 1 --embeddingmodel embed-gemma:300m
```
