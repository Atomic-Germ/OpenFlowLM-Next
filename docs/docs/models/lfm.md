---
title: "LiquidAI LFM"
description: "Compact LiquidAI LFM 2 and LFM 2.5 chat, reasoning, and transcript models on AMD Ryzen AI NPUs."
icon: "waves"
weight: 40
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "llm" "Chat" >}}{{< b "think" "Reasoning" >}}{{< b "tools" "Tools" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b k="hybrid" t="NPU + CPU" tip="Short-convolution projections, gated attention, and FFN layers run on the NPU; short-conv state carry during prompt prefill runs on the CPU." >}}{{< b "gguf" "GGUF" >}}{{< b "q4nx" "Legacy Q4NX" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "1.2B" >}}{{< b "size" "2.6B" >}}{{< b k="unsupported" t="350M" mono="true" >}}{{< b k="unsupported" t="700M" mono="true" >}}{{< b k="unsupported" t="8B-A1B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "32K" >}}{{< b k="qbest" t="Q4_1" mono="true" >}}{{< b k="quant" t="Q4_0" mono="true" >}}{{< b k="quant" t="Q4_K" mono="true" >}}{{< b k="quant" t="Q5_K" mono="true" >}}{{< b k="quant" t="Q6_K" mono="true" >}}{{< b k="quant" t="Q8_0" mono="true" >}}{{< b k="quant" t="IQ4" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Install LFM 2.5 (1.2B) or LFM 2 (1.2B, 2.6B) from Hugging Face GGUF
oflm add LiquidAI/LFM2.5-1.2B-Instruct-GGUF
oflm run lfm2.5-it:1.2b

# Or pull a built-in shortcut
oflm pull lfm2.5-tk:1.2b
oflm run lfm2.5-tk:1.2b
```

Pass `--family` and `--xclbin-from` when installing from a local folder or a custom fine-tune whose name hides the LFM2 architecture:

```bash
oflm add <repo-or-dir> --family lfm2 --xclbin-from LFM2-2.6B-NPU2
```

## Variants & Built-in Shortcuts

| Tag | Variant & Capabilities | Runtime & Format | Context & RAM | Status |
|---|---|---|---|---|
| Custom LFM 2.5 GGUF (`oflm add`) | {{< b "size" "1.2B" >}} {{< b "llm" "Chat" >}} {{< b "think" "Reasoning" >}} {{< b "tools" "Tools" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Short-convolution projections, gated attention, and FFN layers run on the NPU; short-conv state carry during prompt prefill runs on the CPU." >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "32K" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| Custom LFM 2 GGUF (`oflm add`) | {{< b "size" "1.2B" >}} {{< b "size" "2.6B" >}} {{< b "llm" "Chat" >}} {{< b "tools" "Tools" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Short-convolution projections, gated attention, and FFN layers run on the NPU; short-conv state carry during prompt prefill runs on the CPU." >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "32K" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `lfm2.5-it:1.2b` | {{< b "size" "1.2B" >}} {{< b "llm" "Chat" >}} {{< b "think" "Reasoning" >}} {{< b "tools" "Tools" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Short-convolution projections, gated attention, and FFN layers run on the NPU; short-conv state carry during prompt prefill runs on the CPU." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `lfm2.5-tk:1.2b` | {{< b "size" "1.2B" >}} {{< b "llm" "Chat" >}} {{< b "think" "Reasoning" >}} {{< b "tools" "Tools" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Short-convolution projections, gated attention, and FFN layers run on the NPU; short-conv state carry during prompt prefill runs on the CPU." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `lfm2:1.2b` | {{< b "size" "1.2B" >}} {{< b "llm" "Chat" >}} {{< b "tools" "Tools" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Short-convolution projections, gated attention, and FFN layers run on the NPU; short-conv state carry during prompt prefill runs on the CPU." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `lfm2:2.6b` | {{< b "size" "2.6B" >}} {{< b "llm" "Chat" >}} {{< b "tools" "Tools" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Short-convolution projections, gated attention, and FFN layers run on the NPU; short-conv state carry during prompt prefill runs on the CPU." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `lfm2-trans:2.6b` | {{< b "size" "2.6B" >}} {{< b "llm" "Chat" >}} {{< b "tools" "Tools" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Short-convolution projections, gated attention, and FFN layers run on the NPU; short-conv state carry during prompt prefill runs on the CPU." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |

## Tips & Current Limitations

- **LFM 2.5 vs. LFM 2 capabilities:** `LFM 2.5` (`1.2B`) supports chat, `<think>` reasoning, and tool calling; `LFM 2` (`1.2B` and `2.6B`) supports chat and tool calling without `<think>` reasoning (`350M`, `700M`, and `8B-A1B` are not supported).
- **Quantization & loading:** `Q4_1` checkpoints load with zero conversion overhead; OpenFlowLM converts `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `IQ4` GGUF files on first load.
- **Picking a variant:** Use `lfm2.5-it:1.2b` (~1 GB) for fast everyday chat, `lfm2.5-tk:1.2b` for step-by-step reasoning, and `lfm2-trans:2.6b` (~1.8 GB) for summarizing meeting notes or [Whisper](/docs/models/whisper/) transcripts.
- **Context length (`--ctx-len`):** Override the default context window up to `32768` (`32K`) tokens at launch (for example, `oflm serve lfm2.5-it:1.2b --ctx-len 32768`).