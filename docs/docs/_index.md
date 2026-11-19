---
title: "Overview"
description: "Run local LLMs, vision, speech, embeddings, and image generation on AMD Ryzen AI NPUs."
icon: "menu_book"
weight: 1
---

**OpenFlowLM (OFLM)** runs local multimodal AI inference directly from GGUF models on AMD Ryzen AI processors with XDNA2 NPUs, keeping your CPU and GPU free. Install it on **Windows & Linux** to chat in your terminal, generate and edit images, embed documents, transcribe audio, or connect local apps over OpenAI- and Ollama-compatible APIs.

## Getting Started

Check supported Ryzen AI processors and memory guidelines in [Hardware & Quickstart](/docs/install/requirements/), then install OpenFlowLM for your operating system:

| Operating System | Setup Guides | Installation Formats |
|---|---|---|
| **Windows** | [Windows](/docs/install/windows/) | Windows 11 (`x64`) `oflm-setup.msi` installer with AMD Ryzen AI NPU driver `32.0.203.304+` |
| **Linux** | [Linux Overview](/docs/install/linux/) · [Packages](/docs/install/linux/packages/) · [Docker](/docs/install/linux/docker/) · [Nix & NixOS](/docs/install/linux/nix/) | `.deb`, `.rpm`, portable `.tar.gz`, Docker, Nix flake, and NixOS module |

## Documentation Map

<div class="oflm-two-col">
  <div class="oflm-card">

**[Getting Started →](/docs/install/)**

Verify [Hardware & Quickstart](/docs/install/requirements/) requirements and install OpenFlowLM on [Windows](/docs/install/windows/) and [Linux](/docs/install/linux/) ([Packages](/docs/install/linux/packages/), [Docker](/docs/install/linux/docker/), and [Nix & NixOS](/docs/install/linux/nix/)).

  </div>
  <div class="oflm-card">

**[User Guide →](/docs/instructions/)**

Run commands in [CLI Usage](/docs/instructions/cli/), customize [`settings.json`](/docs/instructions/config/), or start [Server Mode](/docs/instructions/server/) for OpenAI/Ollama endpoints, tool calling, Open WebUI, and [Image Generation and Editing](/docs/instructions/server/images/).

  </div>
  <div class="oflm-card">

**[Models →](/docs/models/)**

Check supported model families, GGUF quantizations, and context limits, or browse the interactive [Models](/models/) and [Benchmarks](/benchmarks/) pages.

  </div>
  <div class="oflm-card">

**[Contributing →](/docs/contributing/)**

Compile from [source](/docs/contributing/build/), contribute [runtime code](/docs/contributing/code-contributions/) and [NPU kernels](/docs/contributing/kernel-contributions/), run [tests](/docs/contributing/test-contributions/), and maintain [Python tools](/docs/contributing/tool-contributions/).

  </div>
</div>

## Community

- **GitHub:** [Atomic-Germ/OpenFlowLM-Next](https://github.com/Atomic-Germ/OpenFlowLM-Next)
- **Discord:** [OpenFlowLM Discord](https://discord.gg/AcFAKvVV8c)
