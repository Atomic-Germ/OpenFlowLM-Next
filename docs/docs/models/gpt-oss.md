---
title: "GPT-OSS"
description: "OpenAI GPT-OSS 20B and GPT-OSS Safeguard 20B reasoning models on AMD Ryzen AI NPUs."
icon: "lock"
weight: 140
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "llm" "Chat" >}}{{< b "think" "Reasoning" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b k="closed" t="Closed NPU" tip="Requires OFLM_GPTOSS_ENGINE=closed to run inherited prebuilt NPU kernels on .q4nx shortcuts (omitted on source-built NixOS #244; >32 GB RAM recommended)." >}}{{< b "q4nx" "Legacy Q4NX" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "20B" >}}{{< b k="wip" issue="91" t="20B Open GGUF #91" >}}{{< b k="unsupported" t="120B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "8K" >}}{{< b k="quant" t="4-bit Fixed" mono="true" >}}</div></div></div>
</div>

{{< wip issue="91" title="Open NPU Kernels & GGUF Support in Progress" >}}GPT-OSS 20B has no open NPU kernels on `main` yet (#91, [#240](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/240)), so `oflm add` does not accept GPT-OSS GGUF files. Because OpenFlowLM does not silently fall back to closed engines, pass `OFLM_GPTOSS_ENGINE=closed` to run the `.q4nx` shortcuts on Windows or Linux packages (omitted on source-built NixOS per [#244](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244)).{{< /wip >}}

## Quick Start

```bash
# Download the built-in 20B .q4nx shortcut
oflm pull gpt-oss:20b

# Run on Linux / macOS bash (explicit Closed NPU opt-in required)
OFLM_GPTOSS_ENGINE=closed oflm run gpt-oss:20b
```

```powershell
# Run on Windows PowerShell (explicit Closed NPU opt-in required)
$env:OFLM_GPTOSS_ENGINE="closed"; oflm run gpt-oss:20b

# Inside interactive chat, set reasoning effort (low, medium, or high)
/set r-eff high
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Context & RAM | Status |
|---|---|---|---|---|
| Custom GGUF (`oflm add`) | {{< b "size" "20B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "8K" >}} {{< b "size" "64 GB+ RAM" >}} | {{< b "wip" "91" "WIP #91" >}} {{< b "wip" "240" "GGUF #240" >}} |
| `gpt-oss:20b` | {{< b "size" "20B" >}} | {{< b k="closed" t="Closed NPU" tip="Requires OFLM_GPTOSS_ENGINE=closed on .q4nx only; omitted on source-built NixOS (#244)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "8K" >}} {{< b "size" "64 GB+ RAM" >}} | {{< b "wip" "91" "Open NPU #91" >}} {{< b "wip" "244" "NixOS #244" >}} |
| `gpt-oss-sg:20b` | {{< b "size" "20B" >}} | {{< b k="closed" t="Closed NPU" tip="Requires OFLM_GPTOSS_ENGINE=closed on .q4nx only; omitted on source-built NixOS (#244)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "8K" >}} {{< b "size" "64 GB+ RAM" >}} | {{< b "wip" "91" "Open NPU #91" >}} {{< b "wip" "244" "NixOS #244" >}} |

## Tips & Current Limitations

- **Closed NPU opt-in & NixOS (`#244`):** Both built-in 20B shortcuts download pre-quantized (~14.0 GB on disk; `120B` is not supported) and require `OFLM_GPTOSS_ENGINE=closed`. Source-built NixOS packages omit prebuilt closed NPU binaries ([#244](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244)), so GPT-OSS cannot run on NixOS until #91 lands.
- **System RAM (>32 GB recommended, 64 GB+ ideal):** The OS caps NPU-pinned memory near half of system RAM (~15.6 GB on a 32 GB PC). Because the 20B weights and KV cache occupy ~14 GB+, use a machine with >32 GB (ideally 64 GB+) of RAM, or see `unsloth/Qwen3.6-35B-A3B-GGUF` on the [Qwen](/docs/models/qwen/) page for an open NPU + CPU MoE option.
- **Safeguard variant:** `gpt-oss-sg:20b` runs OpenAI's policy-classification and safety-moderation fine-tune (`openai/gpt-oss-safeguard-20b`) with identical `OFLM_GPTOSS_ENGINE=closed` and `/set r-eff` controls.
