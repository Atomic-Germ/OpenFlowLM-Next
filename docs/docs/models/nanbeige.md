---
title: "Nanbeige"
description: "Nanbeige 4.1 3B chat, reasoning, and tool-calling models on AMD Ryzen AI NPUs."
icon: "explore"
weight: 150
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "llm" "Chat" >}}{{< b "think" "Reasoning" >}}{{< b "tools" "Tools" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "open" "NPU" >}}{{< b "gguf" "GGUF" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "3B" >}}{{< b k="unsupported" t="32B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "256K" >}}{{< b k="qbest" t="Q4_1" mono="true" >}}{{< b k="quant" t="Q4_0" mono="true" >}}{{< b k="quant" t="Q4_K" mono="true" >}}{{< b k="quant" t="Q5_K" mono="true" >}}{{< b k="quant" t="Q6_K" mono="true" >}}{{< b k="quant" t="Q8_0" mono="true" >}}{{< b k="quant" t="IQ4" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Install from Hugging Face GGUF
oflm add bartowski/Nanbeige_Nanbeige4.1-3B-GGUF

# Run interactive chat or serve over HTTP
oflm run nanbeige-nanbeige4.1:3b
oflm serve nanbeige-nanbeige4.1:3b --ctx-len 32768
```

Pass `--family nanbeige` and `--xclbin-from Nanbeige4.1-3B-NPU2` when installing from a local folder or a custom fine-tune so Nanbeige's `<0xXX>` byte-token filter and reasoning/tool parser stay active:

```bash
oflm add <repo-or-dir> --family nanbeige --xclbin-from Nanbeige4.1-3B-NPU2
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Context & RAM | Status |
|---|---|---|---|---|
| Custom GGUF (`oflm add`) | {{< b "size" "3B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "8K / 256K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |

## Tips & Current Limitations

- **Quantization & loading:** `Q4_1` checkpoints load with zero conversion overhead; OpenFlowLM converts `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `IQ4` GGUF files on first load (`32B` is not supported).
- **Long-context window (`--ctx-len`):** Nanbeige 4.1 3B defaults to an `8K` context window for fast startup and low RAM usage; pass `--ctx-len` up to `262144` (`256K`) tokens for large documents or long `<think>` reasoning traces.
- **Tool calling:** Pass OpenAI-compatible function definitions to `POST /v1/chat/completions` in [Server Mode](/docs/instructions/server/tool_calling/).
