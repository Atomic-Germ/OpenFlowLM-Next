---
title: "Llama 3"
description: "Meta Llama 3, Llama 3.1 8B, and Llama 3.2 1B and 3B models on AMD Ryzen AI NPUs."
icon: "hub"
weight: 90
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "llm" "Chat" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "open" "NPU" >}}{{< b "gguf" "GGUF" >}}{{< b "q4nx" "Legacy Q4NX" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "1B" >}}{{< b "size" "3B" >}}{{< b "size" "8B" >}}{{< b k="unsupported" t="11B-Vision" mono="true" >}}{{< b k="unsupported" t="70B" mono="true" >}}{{< b k="unsupported" t="90B" mono="true" >}}{{< b k="unsupported" t="405B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "128K" >}}{{< b k="qbest" t="Q4_1" mono="true" >}}{{< b k="quant" t="Q4_0" mono="true" >}}{{< b k="quant" t="Q4_K" mono="true" >}}{{< b k="quant" t="Q5_K" mono="true" >}}{{< b k="quant" t="Q6_K" mono="true" >}}{{< b k="quant" t="Q8_0" mono="true" >}}{{< b k="quant" t="IQ4" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Install Llama 3.2 (1B, 3B) or Llama 3.0 / 3.1 (8B) from Hugging Face GGUF
oflm add unsloth/Llama-3.2-3B-Instruct-GGUF
oflm add bartowski/Meta-Llama-3.1-8B-Instruct-GGUF

# Run interactive chat or start an API server with a custom context window
oflm run llama3.2:3b
oflm serve llama3.1:8b --ctx-len 32768
```

Pass `--family` and `--xclbin-from` when installing from a local folder or a custom fine-tune whose name hides the Llama 3 architecture:

```bash
oflm add <repo-or-dir> --family llama3 --xclbin-from Llama-3.1-8B-NPU2
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Context & RAM | Status |
|---|---|---|---|---|
| Custom GGUF (`oflm add`) | {{< b "size" "1B" >}} {{< b "size" "3B" >}} {{< b "size" "8B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "128K" >}} {{< b "size" "8–16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `llama3.2:1b` | {{< b "size" "1B" >}} | {{< b "open" "NPU" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "128K" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `llama3.2:3b` | {{< b "size" "3B" >}} | {{< b "open" "NPU" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "64K / 128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `llama3.1:8b` | {{< b "size" "8B" >}} | {{< b "open" "NPU" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "16K / 128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |

## Tips & Current Limitations

- **Supported sizes:** `1B` and `3B` (Llama 3.2) and `8B` (Llama 3.0 and Llama 3.1) run on Open `NPU` (`11B-Vision`, `70B`, `90B`, and `405B` are not supported). `llama3.2:1b` (~1.3 GB) fits alongside an embedding or speech model in minimal RAM, `llama3.2:3b` (~2.7 GB) suits 16 GB laptops, and `llama3.1:8b` (~5.4 GB) handles multi-turn chat and code tasks.
- **Tool-calling templates:** Stock Meta Llama `<|python_tag|>` tool output is not parsed natively by the Llama 3 frontend; community fine-tunes formatted with `<tool_call>` XML tags work with `/v1/chat/completions` tool parsing, or see [Qwen 3](/docs/models/qwen/), [IBM Granite 4](/docs/models/granite/), or [LiquidAI LFM](/docs/models/lfm/) for native tool calling.
- **Quantization & loading:** `Q4_1` checkpoints load with zero conversion overhead; OpenFlowLM converts `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `IQ4` GGUF files on first load.
- **Expanding context (`--ctx-len`):** `llama3.1:8b` defaults to `16K` context and `llama3.2:3b` defaults to `64K` to save RAM; pass `--ctx-len` up to `131072` (`128K`) for long documents (up to `8K` on Llama 3.0 `8B`).