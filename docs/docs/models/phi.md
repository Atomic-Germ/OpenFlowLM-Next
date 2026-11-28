---
title: "Phi"
description: "Microsoft Phi-4-mini 4B chat models on AMD Ryzen AI NPUs."
icon: "bolt"
weight: 100
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "llm" "Chat" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "open" "NPU" >}}{{< b "gguf" "GGUF" >}}{{< b "q4nx" "Legacy Q4NX" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "4B" >}}{{< b k="unsupported" t="3.8B" mono="true" >}}{{< b k="unsupported" t="5.6B Multimodal" mono="true" >}}{{< b k="unsupported" t="14B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "128K" >}}{{< b k="qbest" t="Q4_1" mono="true" >}}{{< b k="quant" t="Q4_0" mono="true" >}}{{< b k="quant" t="Q4_K" mono="true" >}}{{< b k="quant" t="Q5_K" mono="true" >}}{{< b k="quant" t="Q6_K" mono="true" >}}{{< b k="quant" t="Q8_0" mono="true" >}}{{< b k="quant" t="IQ4" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Install Phi-4-mini (4B) from Hugging Face GGUF
oflm add unsloth/Phi-4-mini-instruct-GGUF --family phi4 --xclbin-from Phi4-mini-Instruct-NPU2

# Or pull and run the built-in shortcut
oflm pull phi4-mini-it:4b
oflm run phi4-mini-it:4b
```

Pass `--family phi4 --xclbin-from Phi4-mini-Instruct-NPU2` when installing standalone GGUF repositories or local folders (automatic geometry detection inspects `microsoft/Phi-4-mini-instruct` `config.json` when present):

```bash
oflm add <repo-or-dir> --family phi4 --xclbin-from Phi4-mini-Instruct-NPU2
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Context & RAM | Status |
|---|---|---|---|---|
| Custom Phi-4-mini GGUF (`oflm add`) | {{< b "size" "4B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `phi4-mini-it:4b` | {{< b "size" "4B" >}} | {{< b "open" "NPU" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K / 128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |

## Tips & Current Limitations

- **Phi-4-mini (`4B`) only:** OpenFlowLM compiles open NPU kernels for Phi-4-mini (`4B`). Older Phi-3 and Phi-3.5-mini (`3.8B`), Phi-4-multimodal (`5.6B Multimodal`), and Phi-4 (`14B`) checkpoints use uncompiled geometries and are not supported.
- **Capabilities:** Phi-4-mini supports interactive and HTTP chat (`<|tool_call|>` tool-call parsing is not implemented in the Phi-4 frontend).
- **Quantization & loading:** `Q4_1` checkpoints load with zero conversion overhead; OpenFlowLM converts `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `IQ4` GGUF files on first load.
- **Expanding context (`--ctx-len`):** `phi4-mini-it:4b` (~3.4 GB on disk) defaults to a `32K` context window; pass `--ctx-len` up to `131072` (`128K`) when summarizing long files or codebases.