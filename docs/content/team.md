---
title: "Community & Maintainers"
description: "Get help on Discord, request models on GitHub, share Ryzen AI benchmarks, and meet the OpenFlowLM maintainers."
url: "/team/"
kicker: "Community & Project"
icon: "groups"
ctas:
  - text: "Join Discord"
    url: "https://discord.gg/AcFAKvVV8c"
    icon: "forum"
  - text: "GitHub Repository"
    url: "https://github.com/Atomic-Germ/OpenFlowLM-Next"
    icon: "code"
    outline: true
---

## Independent & Community-Driven

OpenFlowLM is a free, open-source local AI application and NPU runtime built by the community for AMD Ryzen AI PCs. **OpenFlowLM is unaffiliated with and unendorsed by AMD.**

Started after FastFlowLM closed its NPU kernels and model converters, OpenFlowLM keeps the entire stack open so anyone with a Ryzen AI laptop or mini-PC can run standard Hugging Face GGUF models locally without vendor lock-in.

## Get Help & Get Involved

<div class="oflm-two-col">
<div class="oflm-card">
<div class="mb-2">{{< b k="llm" t="Discord & Discussions" icon="forum" >}}</div>
<h3>Ask Questions &amp; Share Setups</h3>
<p class="small mb-2" style="color: var(--oflm-muted);">Join other Ryzen AI laptop and mini-PC owners to compare models, troubleshoot drivers on Windows 11 or Linux, and share Open WebUI, Continue, and Claude Code setups.</p>
<a href="https://discord.gg/AcFAKvVV8c" target="_blank" rel="noopener noreferrer" class="small fw-semibold text-decoration-none">Join the OpenFlowLM Discord &rarr;</a>
</div>

<div class="oflm-card">
<div class="mb-2">{{< b k="open" t="Models & Benchmarks" icon="speed" >}}</div>
<h3>Request Models &amp; Submit Runs</h3>
<p class="small mb-2" style="color: var(--oflm-muted);">Open a GitHub issue to request a new model family or report a bug, or run <code>oflm bench &lt;tag&gt; --submit</code> to add your laptop or mini-PC to the hardware leaderboard.</p>
<div class="d-flex flex-wrap gap-3">

[GitHub Issues →](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues)

[Submit Benchmarks →](/benchmarks/submit/)

</div>
</div>
</div>

## Core Maintainers

<div class="row g-3 my-2 mb-4">
<div class="col-md-4">
<div class="oflm-card h-100">
<div class="d-flex justify-content-between align-items-center mb-2">
<h3 class="h5 fw-bold mb-0">Atomic-Germ</h3>
<span class="oflm-badge open">Founder &amp; Lead</span>
</div>
<p class="small mb-3" style="color: var(--oflm-muted);">Started OpenFlowLM, maintains the <code>oflm add</code> model installer, Qwen 3.5 and Mixture-of-Experts support, and Windows 11 and Linux release installers.</p>
<a href="https://github.com/Atomic-Germ" target="_blank" rel="noopener noreferrer" class="small fw-semibold text-decoration-none">github.com/Atomic-Germ &rarr;</a>
</div>
</div>

<div class="col-md-4">
<div class="oflm-card h-100">
<div class="d-flex justify-content-between align-items-center mb-2">
<h3 class="h5 fw-bold mb-0">vegah</h3>
<span class="oflm-badge open">Models &amp; Runtime</span>
</div>
<p class="small mb-3" style="color: var(--oflm-muted);">Built direct Hugging Face GGUF loading, Gemma 4 and IBM Granite 4 model support, fast prompt processing, and speculative decoding.</p>
<a href="https://github.com/vegah" target="_blank" rel="noopener noreferrer" class="small fw-semibold text-decoration-none">github.com/vegah &rarr;</a>
</div>
</div>

<div class="col-md-4">
<div class="oflm-card h-100">
<div class="d-flex justify-content-between align-items-center mb-2">
<h3 class="h5 fw-bold mb-0">Cyronius</h3>
<span class="oflm-badge open">Image &amp; NPU Runtimes</span>
</div>
<p class="small mb-3" style="color: var(--oflm-muted);">Built NPU image generation and photo editing for FLUX.2 [klein] 4B, plus cross-model accuracy and stability improvements.</p>
<a href="https://github.com/Cyronius" target="_blank" rel="noopener noreferrer" class="small fw-semibold text-decoration-none">github.com/Cyronius &rarr;</a>
</div>
</div>
</div>

## Contributing Code & Open-Source Credits

Developers interested in building OpenFlowLM from source, adding CLI tools, or contributing new NPU kernels can start at the [Contributing Guide](/docs/contributing/).

OpenFlowLM builds on upstream open-source projects across the AMD Ryzen AI and local AI ecosystem:

- **AMD XRT & `amdxdna`:** Open-source host runtime and drivers for AMD Ryzen AI NPUs on Windows and Linux.
- **AMD Open NPU Compiler Tools:** Open-source Python and C++ NPU kernel compiler stack used to build OpenFlowLM's shipped NPU runtimes.
- **GGML / `llama.cpp`:** Standard GGUF model file format supported by `oflm add`.
- **NpuEmbeddings:** Batched NPU text encoder foundation integrated into OpenFlowLM's embedding runtime.
- **FastFlowLM (pre-fork MIT base):** Original CLI and local server foundation; our offer to upstream open NPU runtime work remains open.
