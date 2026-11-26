---
title: "OpenFlowLM"
description: "Fast, open-source inference for chat, vision, embeddings, speech-to-text, and image generation on AMD Ryzen AI NPUs."
echarts: true
---

<div class="pt-3 pb-2">
<div class="d-flex flex-wrap justify-content-between align-items-end gap-3 mb-3">
<div>
<div class="mb-2">{{< b k="open" t="Simple Model Setup" icon="deployed_code" >}}</div>
<h2 class="fw-bold mb-1">Two Ways to Get Models</h2>
<p class="mb-0" style="color: var(--oflm-muted);">OpenFlowLM ships without bundled weights. Add Hugging Face GGUF checkpoints or pull a verified shortcut on demand.</p>
</div>
<div class="d-flex flex-wrap gap-2">
<a href="models/" class="btn btn-sm btn-primary">Explore Models</a>
<a href="docs/install/" class="btn btn-sm btn-outline-primary">Install Guide</a>
</div>
</div>

<div class="oflm-two-col">
<div class="oflm-card">
<div class="d-flex justify-content-between align-items-center flex-wrap gap-2 mb-2">
{{< b k="open" t="Direct GGUF" icon="verified" >}}
<code>oflm add &lt;hf-repo-or-file&gt;</code>
</div>
<h3>Bring Any Compatible GGUF</h3>
<p class="small mb-2" style="color: var(--oflm-muted);">Install supported Hugging Face GGUF repositories, community fine-tunes, or local <code>.gguf</code> files.</p>
<div class="mb-1"><span class="oflm-group-label me-1">Chat &amp; Vision:</span>{{< quants "llm" >}}</div>
<div class="mb-1"><span class="oflm-group-label me-1">Embeddings:</span>{{< quants "embed" >}}</div>
<div class="small mb-2" style="color: var(--oflm-muted);">★ = recommended quantization</div>

```bash
oflm add unsloth/Qwen3.5-4B-GGUF
oflm add unsloth/gemma-3-4b-it-GGUF
```

</div>

<div class="oflm-card">
<div class="d-flex justify-content-between align-items-center flex-wrap gap-2 mb-2">
{{< b k="llm" t="Quick Download" icon="download" >}}
<code>oflm pull &lt;tag&gt;</code>
</div>
<h3>One-Command Shortcuts</h3>
<p class="small mb-2" style="color: var(--oflm-muted);">Pull pre-verified model tags across chat, vision, document embeddings, speech-to-text, and image generation.</p>
<div class="oflm-pill-row mb-1"><span class="oflm-group-label">Capabilities:</span>{{< b "llm" "Chat" >}} {{< b "vlm" "Vision" >}} {{< b "embed" "Embeddings" >}} {{< b "asr" "Speech" >}} {{< b "image" "Image" >}}</div>
<div class="oflm-pill-row mb-2"><span class="oflm-group-label">Formats:</span>{{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} {{< b "bundle" "Pre-packed" >}}</div>

```bash
oflm pull qwen3.5:4b
oflm pull flux2-klein:4b
```

</div>
</div>
</div>

<div class="pt-3 mt-2">
<div class="d-flex flex-wrap justify-content-between align-items-end gap-3 mb-3">
<div>
<div class="mb-2">{{< b k="npu" t="Measured NPU Speeds" icon="speed" >}}</div>
<h2 class="fw-bold mb-1">Measured Throughput by Hardware &amp; Modality</h2>
<p class="mb-0" style="color: var(--oflm-muted);">Faceted per AMD Ryzen AI NPU platform and workload so devices and units are never mixed on the same axis.</p>
</div>
<div class="d-flex flex-wrap align-items-center gap-2">
<a href="benchmarks/" class="btn btn-sm btn-primary">All Benchmarks</a>
<a href="benchmarks/submit/" class="btn btn-sm btn-outline-primary">Submit Run</a>
</div>
</div>

{{< all-runs-chart >}}

{{< bench-chart data="qwen35_2b_ryzen_ai_350" >}}
</div>

<div class="pt-3 mt-2">
<div class="d-flex flex-wrap justify-content-between align-items-end gap-3 mb-4">
<div>
<div class="mb-2">{{< b k="open" t="Everyday Workflows" icon="auto_awesome" >}}</div>
<h2 class="fw-bold mb-1">More Than Terminal Chat</h2>
<p class="mb-0" style="color: var(--oflm-muted);">Inspect screenshots, generate and restyle images offline, or connect desktop AI apps to a local API.</p>
</div>
<div class="d-flex flex-wrap gap-2">
<a href="engines/" class="btn btn-sm btn-outline-primary">NPU Runtimes</a>
<a href="docs/instructions/" class="btn btn-sm btn-outline-primary">Usage Guides</a>
</div>
</div>

<div class="row g-4">
<div class="col-lg-4 col-md-6">
<div class="oflm-card h-100 d-flex flex-column">
<div class="oflm-pill-row mb-2">{{< b k="hybrid" t="NPU + CPU" tip="Language model layers run on the NPU; ViT image encoding runs on the CPU." >}} {{< b "llm" "Chat" >}} {{< b "vlm" "Vision" >}} {{< b "tools" "Tools" >}}</div>
<h3 class="h5 fw-bold mb-2">Vision Q&amp;A</h3>
<p class="small mb-2" style="color: var(--oflm-muted);">Ask questions about screenshots, charts, and photos in the terminal or browser with Qwen3-VL, Qwen 3.5, and Qwen 3.6 MoE.</p>

```bash
oflm run qwen3vl-it:4b
# Inside the chat session:
/input "chart.png" Summarize this chart
```

<a href="docs/models/qwen/" class="small fw-semibold text-decoration-none mt-auto pt-1">Vision model guides &rarr;</a>
</div>
</div>

<div class="col-lg-4 col-md-6">
<div class="oflm-card h-100 d-flex flex-column">
<div class="oflm-pill-row mb-2">{{< b k="open" t="NPU" >}} {{< b "image" "Text-to-Image" >}} {{< b "image" "Image Editing" >}}</div>
<h3 class="h5 fw-bold mb-2">Local Image Generation &amp; Editing</h3>
<p class="small mb-2" style="color: var(--oflm-muted);">Create 512x512 and 1024x1024 images or restyle reference photos offline with FLUX.2 [klein] 4B on your NPU.</p>

```bash
oflm image flux2-klein:4b \
  "a red fox in fresh snow" -o fox.png
oflm image flux2-klein:4b \
  "Make it watercolor" --image fox.png
```

<a href="docs/instructions/server/images/" class="small fw-semibold text-decoration-none mt-auto pt-1">Image generation guide &rarr;</a>
</div>
</div>

<div class="col-lg-4 col-md-6">
<div class="oflm-card h-100 d-flex flex-column">
<div class="oflm-pill-row mb-2"><code>oflm serve</code> {{< b "llm" "Chat" >}} {{< b "embed" "Embeddings" >}} {{< b "asr" "Speech" >}} {{< b "tools" "Tools" >}}</div>
<h3 class="h5 fw-bold mb-2">Local OpenAI &amp; Anthropic API</h3>
<p class="small mb-2" style="color: var(--oflm-muted);">Connect Open WebUI, Continue, Claude Code, and OpenAI SDKs to local OpenAI and Anthropic API endpoints.</p>

```bash
oflm serve qwen3.5:4b \
  --embed 1 --embeddingmodel bge-base:en-v1.5 \
  --asr 1
```

<a href="docs/instructions/server/" class="small fw-semibold text-decoration-none mt-auto pt-1">Server setup guides &rarr;</a>
</div>
</div>
</div>
</div>
