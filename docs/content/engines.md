---
title: "NPU Runtimes & Acceleration"
description: "See how OpenFlowLM runs chat, vision, embeddings, speech, and image models across your AMD Ryzen AI NPU and CPU."
url: "/engines/"
kicker: "How It Runs"
icon: "memory"
ctas:
  - text: "Browse Models"
    url: "/models/"
    icon: "view_module"
  - text: "View Benchmarks"
    url: "/benchmarks/"
    icon: "speed"
    outline: true
---

{{< engines-matrix >}}

## What the Acceleration Labels Mean

<div class="oflm-two-col">
<div class="oflm-card">
<div class="mb-2">{{< b k="open" t="NPU" >}}</div>
<h3>Open NPU Kernels &amp; Quietest Fans</h3>
<p class="small mb-0" style="color: var(--oflm-muted);">Compiled from OpenFlowLM's open-source NPU kernels in <code>open_kernels/</code>. Heavy neural network layers run on the Ryzen AI NPU while your CPU cores and integrated Radeon GPU stay free.</p>
</div>

<div class="oflm-card">
<div class="mb-2">{{< b k="hybrid" t="NPU + CPU" tip="Heavy attention and matrix projections run on the NPU; lightweight routing, ViT image encoding, token decoding, or prompt state updates run on the CPU." >}}</div>
<h3>Heavy Math on NPU, Light Helpers on CPU</h3>
<p class="small mb-0" style="color: var(--oflm-muted);">Open NPU kernels execute matrix multiplications and attention layers while compact host stages (Mixture-of-Experts routing, Qwen ViT image encoding, or recurrent state carry on long prompts) run on the CPU.</p>
</div>
</div>

<div class="oflm-card mt-3">
<div class="mb-2">{{< b k="closed" t="Closed NPU" tip="Inherited prebuilt NPU kernels and closed runtime libraries; fixed .q4nx shortcuts only and omitted from source-built NixOS packages (#244)." >}}</div>
<h3>Inherited Prebuilt Kernels (Being Phased Out)</h3>
<p class="small mb-0" style="color: var(--oflm-muted);">A small set of unported <code>.q4nx</code> shortcuts (<code>gpt-oss:20b</code>, <code>gemma4-it:e2b/e4b</code>, <code>whisper-v3:turbo</code>, <code>qwen2.5vl-it:3b</code>, and Gemma vision) still use inherited prebuilt NPU binaries and closed runtime libraries. OpenFlowLM never silently swaps to a closed engine when open kernels are missing: most families require an explicit <code>OFLM_&lt;FAMILY&gt;_ENGINE=closed</code> opt-in, closed engines cannot load GGUF files, and prebuilt closed binaries are omitted from source-built NixOS packages (<a href="https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244">#244</a>).</p>
</div>

### Tips for Laptop & Mini-PC Users

- **Unified system memory:** Ryzen AI 300 and Ryzen AI Max processors share system RAM with the NPU. Models from 4B to 8B fit well on 16 GB systems; 27B and 35B-A3B models work best with 32 GB to 64 GB+ of RAM.
- **Run chat, embeddings, and speech together:** `oflm serve` keeps a chat model alongside a batched embedding model (`--embed 1`) or Whisper speech-to-text (`--asr 1`) for Open WebUI, Continue, Claude Code, and OpenAI SDKs.
- **Windows, Linux & NixOS packages:** Open NPU kernels ship in the Windows MSI installer, Linux DEB/RPM/TGZ releases, and NixOS (`.#oflm`, or `.#oflm-with-bert` for BERT embeddings); source-built NixOS omits prebuilt closed binaries (`#244`).
- **Open vs. closed `.q4nx` shortcuts:** Most `.q4nx` shortcuts (`qwen3*`, `qwen3.5*`, `qwen3.6-moe*`, `llama3*`, `lfm2*`, `phi4-mini-it:4b`, `deepseek-r1*`, and text on `gemma3:4b`/`gemma4-it:12b`) run on open NPU kernels, while unported shortcuts use `Closed NPU` until open replacements finish.

Building from source or writing NPU kernels? Read the [Contributing Guide](/docs/contributing/).
