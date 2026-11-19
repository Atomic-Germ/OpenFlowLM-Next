---
title: "MedGemma"
description: "Medical research text and multimodal chat with Google MedGemma 4B and MedGemma 1.5 4B on AMD Ryzen AI NPUs."
icon: "medical_services"
weight: 110
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "llm" "Chat" >}}{{< b "vlm" "Vision" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "open" "NPU" >}}{{< b k="closed" t="Closed NPU" tip="Built-in medgemma:4b and medgemma1.5:4b .q4nx shortcuts require OFLM_GEMMA_ENGINE=closed (lacks open_kernels symlink and uses closed gemma_npu for vision; omitted on source-built NixOS #244)." >}}{{< b "gguf" "GGUF" >}}{{< b "q4nx" "Legacy Q4NX" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "4B" >}}{{< b k="unsupported" t="27B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "128K" >}}{{< b k="qbest" t="Q4_1" mono="true" >}}{{< b k="quant" t="Q4_0" mono="true" >}}{{< b k="quant" t="Q4_K" mono="true" >}}{{< b k="quant" t="Q5_K" mono="true" >}}{{< b k="quant" t="Q6_K" mono="true" >}}{{< b k="quant" t="Q8_0" mono="true" >}}{{< b k="quant" t="IQ4" mono="true" >}}</div></div></div>
</div>

> [!WARNING]
> **Research Disclaimer:** Google designed MedGemma for medical research and educational exploration, not clinical diagnosis or patient care ([arXiv:2507.05201](https://arxiv.org/abs/2507.05201)).

## Quick Start

```bash
# Install from Hugging Face GGUF for Open NPU text chat (shares Gemma 3 4B open kernels)
oflm add unsloth/medgemma-4b-it-GGUF
oflm add unsloth/medgemma-1.5-4b-it-GGUF

# Or run a built-in .q4nx shortcut for medical image attachments (Closed NPU opt-in required)
oflm pull medgemma:4b
OFLM_GEMMA_ENGINE=closed oflm run medgemma:4b
```

Pass `--family` and `--xclbin-from` when installing from a local folder or a custom medical fine-tune built on [Gemma 3 4B](/docs/models/gemma/):

```bash
oflm add <repo-or-dir> --family gemma3 --xclbin-from Gemma3-4B-NPU2
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Context & RAM | Status |
|---|---|---|---|---|
| Custom GGUF (`oflm add`) | {{< b "size" "4B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "73" "Vision #73" >}} |
| `medgemma:4b` | {{< b "size" "4B" >}} | {{< b k="closed" t="Closed NPU" tip="Lacks open_kernels symlink to Gemma3-4B-NPU2 and uses closed gemma_npu for images; requires OFLM_GEMMA_ENGINE=closed (omitted on source-built NixOS #244)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "64K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "wip" "73" "Vision #73" >}} {{< b "wip" "244" "NixOS #244" >}} |
| `medgemma1.5:4b` | {{< b "size" "4B" >}} | {{< b k="closed" t="Closed NPU" tip="Lacks open_kernels symlink to Gemma3-4B-NPU2 and uses closed gemma_npu for images; requires OFLM_GEMMA_ENGINE=closed (omitted on source-built NixOS #244)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "64K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "wip" "73" "Vision #73" >}} {{< b "wip" "244" "NixOS #244" >}} |

## Tips & Current Limitations

{{< wip issue="73" title="Open Multimodal Vision Support" >}}GGUF MedGemma checkpoints share the Gemma 3 `4B` open text kernels and run text only while issue #73 adds open vision support.{{< /wip >}}

- **Open NPU GGUF text vs. Closed NPU `.q4nx` shortcuts:** Installing `unsloth/medgemma-4b-it-GGUF` or `unsloth/medgemma-1.5-4b-it-GGUF` via `oflm add` runs text chat on Open `NPU` (`27B` is not supported). Built-in `.q4nx` shortcuts (`medgemma:4b` and `medgemma1.5:4b`) lack `open_kernels` directory symlinks to `Gemma3-4B-NPU2` and require the closed `gemma_npu` library for images; launch them with `OFLM_GEMMA_ENGINE=closed` (closed prebuilts are omitted on source-built NixOS per [#244](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244)).
- **Quantization & loading:** `Q4_1` checkpoints load with zero conversion overhead; OpenFlowLM converts `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `IQ4` GGUF files on first load.
- **Context length (`--ctx-len`):** Pass `--ctx-len` up to `131072` (`128K`) on GGUF installs (`64K` default on `.q4nx` shortcuts) for long research papers or clinical notes.