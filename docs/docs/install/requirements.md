---
title: "Hardware & Quickstart"
description: "Supported AMD Ryzen AI XDNA2 processors, memory guidelines, and three-step post-install quickstart."
icon: "memory"
weight: 5
---

OpenFlowLM runs on **Windows & Linux** systems equipped with an AMD Ryzen AI processor featuring an XDNA2 NPU.

| Requirement | Specification |
|---|---|
| **NPU Architecture** | AMD XDNA2 (Ryzen AI `50+ TOPS` NPU) |
| **Minimum RAM** | `16 GB` (`1B` to `4B` models) |
| **Recommended RAM** | `32 GB+` (`8B+` LLMs, Mixture-of-Experts models, long context windows, and image generation) |
| **Operating Systems** | Windows & Linux (`x86_64`) |

## Processor Compatibility

| Processor Family | Codename | NPU Generation | Status |
|---|---|---|---|
| Ryzen AI Max 300-series | Strix Halo | XDNA2 | {{< b "ok" "Supported" >}} |
| Ryzen AI 300-series | Strix Point, Kraken Point | XDNA2 | {{< b "ok" "Supported" >}} |
| Ryzen AI 400-series | Gorgon Point | XDNA2 | {{< b "ok" "Supported" >}} |
| Ryzen Z2 Extreme | Handheld APUs | XDNA2 | {{< b "ok" "Supported" >}} |
| Ryzen 7040 / 8040 / 200-series | Phoenix, Hawk Point | XDNA1 | {{< b "no" "Unsupported" >}} |

## Operating System Support

| Operating System | Supported Environments & Packages | Install Guides |
|---|---|---|
| **Windows** | Windows 11 (`x64`, 24H2 recommended) with AMD Ryzen AI NPU driver `32.0.203.304+` and `oflm-setup.msi` | [Windows](/docs/install/windows/) |
| **Linux** | Ubuntu `.deb`, Fedora `.rpm`, Arch & portable `.tar.gz`, Docker, and Nix / NixOS (`x86_64`) | [Linux Overview](/docs/install/linux/) · [Packages](/docs/install/linux/packages/) · [Docker](/docs/install/linux/docker/) · [Nix & NixOS](/docs/install/linux/nix/) |

## 3-Step Quickstart {#quickstart}

### 1. Validate Your NPU {#step-1-validate-your-npu}

Confirm that your NPU driver and OpenFlowLM runtime detect your Ryzen AI processor:

```bash
oflm validate
```

### 2. Add a GGUF Model or Pull a Shortcut Tag {#step-2-add-or-pull-a-model}

<div class="oflm-two-col">
  <div class="oflm-card">

#### Direct GGUF (`oflm add`)

Download any compatible GGUF repository from Hugging Face and register it for NPU execution:

```bash
oflm add unsloth/Qwen3.5-4B-GGUF
```

  </div>
  <div class="oflm-card">

#### Built-in Tag (`oflm pull`)

Download a curated shortcut tag from the built-in model list:

```bash
oflm pull qwen3.5:4b
# Or a lightweight 1B model: oflm pull llama3.2:1b
```

  </div>
</div>

### 3. Serve or Chat {#step-3-serve-or-chat}

Start the local OpenAI- and Ollama-compatible server on `http://127.0.0.1:52625`, or chat in your terminal:

```bash
oflm serve qwen3.5:4b
# Or interactive terminal chat:
oflm run qwen3.5:4b
```

Next, explore [CLI Usage](/docs/instructions/cli/), [Configuration](/docs/instructions/config/), [Server Mode](/docs/instructions/server/), or [Image Generation and Editing](/docs/instructions/server/images/).
