---
title: "TranslateGemma"
description: "Multilingual local translation with Google TranslateGemma 4B and 12B on AMD Ryzen AI NPUs."
icon: "translate"
weight: 120
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "llm" "Chat" >}}{{< b "vlm" "Vision" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "open" "NPU" >}}{{< b k="closed" t="Closed NPU" tip="Built-in translategemma:4b .q4nx shortcut requires OFLM_GEMMA_ENGINE=closed (lacks open_kernels symlink and uses closed gemma_npu for vision; omitted on source-built NixOS #244)." >}}{{< b "gguf" "GGUF" >}}{{< b "q4nx" "Legacy Q4NX" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "4B" >}}{{< b "size" "12B" >}}{{< b k="unsupported" t="27B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "128K" >}}{{< b k="qbest" t="Q4_1" mono="true" >}}{{< b k="quant" t="Q4_0" mono="true" >}}{{< b k="quant" t="Q4_K" mono="true" >}}{{< b k="quant" t="Q5_K" mono="true" >}}{{< b k="quant" t="Q6_K" mono="true" >}}{{< b k="quant" t="Q8_0" mono="true" >}}{{< b k="quant" t="IQ4" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Install 4B or 12B from Hugging Face GGUF for Open NPU text translation
oflm add unsloth/translategemma-4b-it-GGUF
oflm add unsloth/translategemma-12b-it-GGUF

# Or run the built-in 4B .q4nx shortcut for image translation (Closed NPU opt-in required)
oflm pull translategemma:4b
OFLM_GEMMA_ENGINE=closed oflm run translategemma:4b
```

Pass `--family` and `--xclbin-from` when installing from a local folder or a custom translation fine-tune built on [Gemma 3](/docs/models/gemma/):

```bash
oflm add <repo-or-dir> --family gemma3 --xclbin-from Gemma3-4B-NPU2
```

Send a single user message formatted with Google's prompt template and **two blank lines** before `{TEXT}` ([arXiv:2601.09012](https://arxiv.org/pdf/2601.09012)):

```text
You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.
Produce only the {TARGET_LANG} translation, without any additional explanations or commentary. Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:


{TEXT}
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Context & RAM | Status |
|---|---|---|---|---|
| Custom GGUF (`oflm add`) | {{< b "size" "4B" >}} {{< b "size" "12B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "128K" >}} {{< b "size" "16–32 GB+ RAM" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "73" "Vision #73" >}} |
| `translategemma:4b` | {{< b "size" "4B" >}} | {{< b k="closed" t="Closed NPU" tip="Lacks open_kernels symlink to Gemma3-4B-NPU2 and uses closed gemma_npu for images; requires OFLM_GEMMA_ENGINE=closed (omitted on source-built NixOS #244)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "64K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "wip" "73" "Vision #73" >}} {{< b "wip" "244" "NixOS #244" >}} |

## Tips & Current Limitations

{{< wip issue="73" title="Open Multimodal Image Translation" >}}GGUF TranslateGemma checkpoints (`4B` and `12B`) share the Gemma 3 open text kernels and run text translation on Open `NPU` while issue #73 adds open vision support.{{< /wip >}}

- **Open NPU GGUF text vs. Closed NPU `.q4nx` shortcut:** Installing `unsloth/translategemma-4b-it-GGUF` or `unsloth/translategemma-12b-it-GGUF` via `oflm add` runs text translation 100% on Open `NPU` (`27B` is not supported). The built-in `translategemma:4b` `.q4nx` shortcut lacks an `open_kernels` symlink to `Gemma3-4B-NPU2` and needs the closed `gemma_npu` library for images; launch it with `OFLM_GEMMA_ENGINE=closed` (closed prebuilts are omitted on source-built NixOS per [#244](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244)).
- **Quantization & loading:** `Q4_1` checkpoints load with zero conversion overhead; OpenFlowLM converts `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `IQ4` GGUF files on first load.
- **Long document translation (`--ctx-len`):** Pass `--ctx-len` up to `131072` (`128K`) on GGUF installs (`64K` default on `translategemma:4b`) for book chapters or long articles.