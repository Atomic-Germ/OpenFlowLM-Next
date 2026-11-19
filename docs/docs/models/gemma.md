---
title: "Gemma"
description: "Google Gemma 3 and Gemma 4 chat and multimodal models on AMD Ryzen AI NPUs."
icon: "auto_awesome"
weight: 20
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "llm" "Chat" >}}{{< b "vlm" "Vision" >}}{{< b "asr" "Speech" >}}{{< b "think" "Reasoning" >}}{{< b "tools" "Tools" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "open" "NPU" >}}{{< b k="closed" t="Closed NPU" tip="Gemma 4 E2B/E4B .q4nx shortcuts and vision/audio on gemma4-it:12b or gemma3:4b run on inherited prebuilt kernels (OFLM_GEMMA4_ENGINE=closed / OFLM_GEMMA_ENGINE=closed; omitted on source-built NixOS #244)." >}}{{< b "gguf" "GGUF" >}}{{< b "q4nx" "Legacy Q4NX" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "4B" >}}{{< b "size" "12B" >}}{{< b k="wip" issue="236" t="E2B #236" >}}{{< b k="wip" issue="236" t="E4B #236" >}}{{< b k="unsupported" t="270M" mono="true" >}}{{< b k="unsupported" t="1B" mono="true" >}}{{< b k="unsupported" t="26B-A4B" mono="true" >}}{{< b k="unsupported" t="27B" mono="true" >}}{{< b k="unsupported" t="31B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "128K" >}}{{< b k="qbest" t="Q4_1" mono="true" >}}{{< b k="quant" t="Q4_0" mono="true" >}}{{< b k="quant" t="Q4_K" mono="true" >}}{{< b k="quant" t="Q5_K" mono="true" >}}{{< b k="quant" t="Q6_K" mono="true" >}}{{< b k="quant" t="Q8_0" mono="true" >}}{{< b k="quant" t="IQ4" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Install Gemma 3 (4B, 12B) from Hugging Face GGUF for Open NPU text chat
oflm add unsloth/gemma-3-4b-it-GGUF
oflm add unsloth/gemma-3-12b-it-GGUF

# Or pull and run the Gemma 4 12B shortcut on Open NPU (text)
oflm pull gemma4-it:12b
oflm run gemma4-it:12b
```

Pass `--family` and `--xclbin-from` when installing a compatible Gemma 3 (`4B`/`12B`) or Gemma 4 (`12B`) GGUF from a local folder or custom repository:

```bash
oflm add <repo-or-dir> --family gemma3 --xclbin-from Gemma3-4B-NPU2
oflm add <repo-or-dir> --family gemma4 --xclbin-from Gemma4-12B-IT-NPU2
```

Attach images or audio files via `/input` in [CLI Usage](/docs/instructions/cli/) using the temporary `Closed NPU` `.q4nx` shortcuts (Windows and Linux packages; omitted on source-built NixOS `#244`):

```bash
# Gemma 4 E2B / E4B (.q4nx shortcuts run text, vision, and audio on Closed NPU)
oflm pull gemma4-it:e2b
oflm run gemma4-it:e2b

# Or opt into Closed NPU vision/audio on gemma4-it:12b or vision on gemma3:4b
OFLM_GEMMA4_ENGINE=closed oflm run gemma4-it:12b
OFLM_GEMMA_ENGINE=closed oflm run gemma3:4b

# Inside interactive chat:
/input "path/to/image.jpg" describe this image.
/input "path/to/audio.mp3" summarize this audio.
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Context & RAM | Status |
|---|---|---|---|---|
| Custom Gemma 3 (`4B`, `12B`) / Gemma 4 (`12B`) GGUF (`oflm add`) | {{< b "size" "4B" >}} {{< b "size" "12B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "128K" >}} {{< b "size" "16–32 GB+ RAM" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "73" "Vision #73" >}} |
| Custom Gemma 4 `E2B` / `E4B` GGUF (`oflm add`) | {{< b "size" "E2B" >}} {{< b "size" "E4B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "32K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "wip" "236" "WIP #236" >}} {{< b "wip" "73" "Multimodal #73" >}} |
| `gemma3:4b` | {{< b "size" "4B" >}} | {{< b "open" "NPU" >}} {{< b k="closed" t="Closed NPU" tip="Text runs on Open NPU; attaching images on gemma3:4b requires OFLM_GEMMA_ENGINE=closed (omitted on source-built NixOS #244)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "64K / 128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "73" "Vision #73" >}} |
| `gemma4-it:12b` | {{< b "size" "12B" >}} | {{< b "open" "NPU" >}} {{< b k="closed" t="Closed NPU" tip="Text decoder runs on Open NPU; attaching images or audio on gemma4-it:12b requires OFLM_GEMMA4_ENGINE=closed (omitted on source-built NixOS #244)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K / 128K" >}} {{< b "size" "32 GB+ RAM" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "73" "Vision/Audio #73" >}} |
| `gemma4-it:e2b` | {{< b "size" "E2B" >}} | {{< b k="closed" t="Closed NPU" tip="Runs text, vision, and audio via closed gemma4e_npu on .q4nx only; omitted on source-built NixOS (#244)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "wip" "236" "Open NPU #236" >}} {{< b "wip" "244" "NixOS #244" >}} |
| `gemma4-it:e4b` | {{< b "size" "E4B" >}} | {{< b k="closed" t="Closed NPU" tip="Runs text, vision, and audio via closed gemma4e_npu on .q4nx only; omitted on source-built NixOS (#244)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "wip" "236" "Open NPU #236" >}} {{< b "wip" "244" "NixOS #244" >}} |
| `gemma3:1b` | {{< b k="unsupported" t="1B" mono="true" >}} | {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b k="unsupported" t="Unsupported" >}} |

## Tips & Current Limitations

{{< wip issue="236" title="Gemma 4 E2B/E4B Open Kernels & Multimodal Support" >}}Issue #236 tracks open NPU text kernels and GGUF loading for Gemma 4 `E2B` and `E4B` (`oflm add` currently refuses `E2B` and `E4B` GGUFs), and [#73](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/73) tracks open multimodal vision and audio across Gemma.{{< /wip >}}

- **Unsupported Gemma variants:** Gemma 4 `26B-A4B` and `31B`, as well as Gemma 3 `270M`, `1B`, and `27B`, are not supported. Although `gemma3:1b` still appears in `model_list.json`, the closed `gemma3-text` engine was removed and no `1B` open kernel set exists, so `gemma3:1b` is not runnable in current builds; use [Llama 3.2 1B](/docs/models/llama/), [Qwen 3 0.6B/1.7B](/docs/models/qwen/), or [LiquidAI LFM 1.2B](/docs/models/lfm/) for sub-2B workloads.
- **Image & audio attachments (`Closed NPU`):** GGUF Gemma checkpoints run text only. Until [#73](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/73) lands, multimodal inputs run on `.q4nx` shortcuts via `gemma4-it:e2b` / `gemma4-it:e4b` (`gemma4e_npu`), `OFLM_GEMMA4_ENGINE=closed` on `gemma4-it:12b`, or `OFLM_GEMMA_ENGINE=closed` on `gemma3:4b` (all closed prebuilts are omitted on source-built NixOS per [#244](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244)).
- **Quantization & loading:** `Q4_1` checkpoints load with zero conversion overhead; OpenFlowLM converts `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `IQ4` GGUF files on first load.
- **Domain fine-tunes:** See [MedGemma](/docs/models/medgemma/) and [TranslateGemma](/docs/models/translategemma/) for medical and translation models sharing the Gemma 3 `4B` and `12B` open text kernels.
