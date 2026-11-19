---
title: "DeepSeek"
description: "DeepSeek-R1 8B reasoning distillations on AMD Ryzen AI NPUs."
icon: "psychology_alt"
weight: 130
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "llm" "Chat" >}}{{< b "think" "Reasoning" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "open" "NPU" >}}{{< b "gguf" "GGUF" >}}{{< b "q4nx" "Legacy Q4NX" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "8B" >}}{{< b k="unsupported" t="1.5B" mono="true" >}}{{< b k="unsupported" t="7B" mono="true" >}}{{< b k="unsupported" t="14B" mono="true" >}}{{< b k="unsupported" t="32B" mono="true" >}}{{< b k="unsupported" t="70B" mono="true" >}}{{< b k="unsupported" t="671B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "128K" >}}{{< b k="qbest" t="Q4_1" mono="true" >}}{{< b k="quant" t="Q4_0" mono="true" >}}{{< b k="quant" t="Q4_K" mono="true" >}}{{< b k="quant" t="Q5_K" mono="true" >}}{{< b k="quant" t="Q6_K" mono="true" >}}{{< b k="quant" t="Q8_0" mono="true" >}}{{< b k="quant" t="IQ4" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Install 8B DeepSeek-R1 distillations (Qwen3-8B or Llama-3.1-8B base) from Hugging Face GGUF
oflm add unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF
oflm add unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF

# Or pull and run a built-in 8B shortcut
oflm pull deepseek-r1-0528:8b
oflm run deepseek-r1-0528:8b
```

Pass `--family` and `--xclbin-from` when installing from a local folder or a custom 8B fine-tune whose name hides whether the base model is [Llama 3.1 8B](/docs/models/llama/) or [Qwen 3 8B](/docs/models/qwen/):

```bash
oflm add <repo-or-dir> --family qwen3 --xclbin-from Qwen3-8B-NPU2
oflm add <repo-or-dir> --family llama3 --xclbin-from Llama-3.1-8B-NPU2
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Context & RAM | Status |
|---|---|---|---|---|
| Custom GGUF (`oflm add`) | {{< b "size" "8B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `deepseek-r1:8b` | {{< b "size" "8B" >}} | {{< b "open" "NPU" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "16K / 128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `deepseek-r1-0528:8b` | {{< b "size" "8B" >}} | {{< b "open" "NPU" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "16K / 128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |

## Tips & Current Limitations

- **Supported distillations (8B only):** OpenFlowLM compiles dense open NPU kernels for `8B` distillations built on [Qwen 3 8B](/docs/models/qwen/) (`deepseek-r1-0528:8b`, ~5.6 GB) and [Llama 3.1 8B](/docs/models/llama/) (`deepseek-r1:8b`, ~5.4 GB). Upstream `1.5B`, `7B`, `14B`, `32B`, `70B`, and `671B` checkpoints use uncompiled Qwen 2.5 or DeepSeek-V3 MoE geometries and are not supported.
- **Quantization & NixOS note:** `Q4_1` GGUF checkpoints load with zero conversion overhead; OpenFlowLM converts `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `IQ4` GGUF files on first load. On source-built NixOS where legacy `src/xclbins` derivative symlinks are omitted ([#244](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244)), install the GGUF repositories above via `oflm add`.
- **Room for `<think>` chains (`--ctx-len`):** Reasoning models emit `<think>...</think>` blocks before the final answer. Both shortcuts default to `16K` context; pass `--ctx-len 32768` (up to `131072`) on multi-step math and code tasks to avoid truncating long traces.