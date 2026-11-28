---
title: "IBM Granite"
description: "Dense IBM Granite 4 3B chat, reasoning, and tool-calling models on AMD Ryzen AI NPUs."
icon: "memory"
weight: 30
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "llm" "Chat" >}}{{< b "think" "Reasoning" >}}{{< b "tools" "Tools" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "open" "NPU" >}}{{< b "gguf" "GGUF" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "3B" >}}{{< b k="unsupported" t="350M" mono="true" >}}{{< b k="unsupported" t="1B" mono="true" >}}{{< b k="unsupported" t="4.0-H Hybrid" mono="true" >}}{{< b k="unsupported" t="7B-A1B" mono="true" >}}{{< b k="unsupported" t="8B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "128K" >}}{{< b k="qbest" t="Q4_1" mono="true" >}}{{< b k="quant" t="Q4_0" mono="true" >}}{{< b k="quant" t="Q4_K" mono="true" >}}{{< b k="quant" t="Q5_K" mono="true" >}}{{< b k="quant" t="Q6_K" mono="true" >}}{{< b k="quant" t="Q8_0" mono="true" >}}{{< b k="quant" t="IQ4" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Install a dense 40-layer Granite 4.0 Micro or Granite 4.2 3B GGUF from Hugging Face
oflm add ibm-granite/granite-4.0-micro-GGUF

# Launch interactive chat or API server
oflm run <granite-tag>
oflm serve <granite-tag>
```

Pass `--family` and `--xclbin-from` when installing from a local folder or a custom dense 3B fine-tune whose name hides the Granite architecture:

```bash
oflm add <repo-or-dir> --family granite --xclbin-from Granite-4.2-3B-NPU2
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Context & RAM | Status |
|---|---|---|---|---|
| Custom Dense 3B GGUF (`oflm add`) | {{< b "size" "3B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "240" "Shortcut #240" >}} |

## Tips & Current Limitations

{{< wip issue="240" title="Built-in Shortcut & Speculative Drafter" >}}Issue #240 tracks adding a built-in Granite 4.2 `3B` `oflm pull` shortcut tag, and [#189](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/189) tracks the Uno speculative drafter on NPU.{{< /wip >}}

- **Dense 3B only (no Mamba2 Hybrid):** OpenFlowLM supports **dense** 40-layer Granite 4.0 Micro (`ibm-granite/granite-4.0-micro-GGUF`) and Granite 4.2 `3B` checkpoints on Open `NPU`. Granite 4.0 **Hybrid Mamba2** models (`granite-4.0-h-*`, `350M`, `1B`, `7B-A1B`) and `8B` checkpoints are not supported.
- **Quantization & loading:** `Q4_1` checkpoints load with zero conversion overhead; OpenFlowLM converts `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `IQ4` GGUF files on first load.
- **Faster prompt processing (`--prefill-mode bfp16`):** Opt into `bfp16` block prefill (~730 to 850 tok/s on Ryzen AI 300 NPUs) for long documents or RAG prompts:

```bash
oflm run <granite-tag> --prefill-mode bfp16
```

- **Lossless speculative decoding (`--speculative`):** Enable prompt-lookup speculative decoding for greedy (`temperature = 0`) code editing, summarization, and structured JSON requests (2.4x to 3.1x faster generation):

```bash
oflm serve <granite-tag> --speculative
```

- **Skipping reasoning traces:** Pass `"think": false` in `/v1/chat/completions` JSON requests to suppress `<think>...</think>` blocks.
