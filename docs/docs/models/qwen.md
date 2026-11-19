---
title: "Qwen"
description: "Qwen 3, Qwen 3.5, Qwen 3.6 MoE, Qwen 2.5 3B, and Qwen-VL models on AMD Ryzen AI NPUs."
icon: "psychology"
weight: 10
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "llm" "Chat" >}}{{< b "vlm" "Vision" >}}{{< b "think" "Reasoning" >}}{{< b "tools" "Tools" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "open" "NPU" >}}{{< b k="hybrid" t="NPU + CPU" tip="Qwen 3.5 and Qwen 3.6 MoE run attention and FFN/expert projections on the NPU; MoE routing, multi-token prompt recurrent state updates, and ViT image encoding (across Qwen-VL, Qwen 3.5, and Qwen 3.6 MoE) run on the CPU." >}}{{< b k="closed" t="Closed NPU" tip="Only the legacy qwen2.5vl-it:3b .q4nx shortcut requires OFLM_QWEN_ENGINE=closed (omitted on NixOS #244); Qwen 2.5-VL 3B GGUFs run on the open runtime." >}}{{< b "gguf" "GGUF" >}}{{< b "q4nx" "Legacy Q4NX" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "0.6B" >}}{{< b "size" "0.8B" >}}{{< b "size" "1.7B" >}}{{< b "size" "2B" >}}{{< b "size" "3B (Qwen 2.5)" >}}{{< b "size" "4B" >}}{{< b "size" "VL 4B" >}}{{< b "size" "8B" >}}{{< b "size" "VL 8B" >}}{{< b "size" "9B" >}}{{< b "size" "14B" >}}{{< b "size" "27B" >}}{{< b "size" "35B-A3B" >}}{{< b k="unsupported" t="0.5B" mono="true" >}}{{< b k="unsupported" t="1.5B" mono="true" >}}{{< b k="unsupported" t="7B" mono="true" >}}{{< b k="unsupported" t="14B (Qwen 2.5)" mono="true" >}}{{< b k="unsupported" t="30B-A3B" mono="true" >}}{{< b k="unsupported" t="32B" mono="true" >}}{{< b k="unsupported" t="72B" mono="true" >}}{{< b k="unsupported" t="122B-A10B" mono="true" >}}{{< b k="unsupported" t="235B-A22B" mono="true" >}}{{< b k="unsupported" t="397B-A17B" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "256K" >}}{{< b k="qbest" t="Q4_1" mono="true" >}}{{< b k="quant" t="Q4_0" mono="true" >}}{{< b k="quant" t="Q4_K" mono="true" >}}{{< b k="quant" t="Q5_K" mono="true" >}}{{< b k="quant" t="Q6_K" mono="true" >}}{{< b k="quant" t="Q8_0" mono="true" >}}{{< b k="quant" t="IQ4" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Install any compatible Hugging Face GGUF (auto-downloads mmproj*.gguf for vision repos)
oflm add unsloth/Qwen3-4B-GGUF
oflm add unsloth/Qwen3.5-9B-GGUF
oflm add unsloth/Qwen3.5-27B-GGUF
oflm add unsloth/Qwen3.6-35B-A3B-GGUF

# Or pull and run a built-in shortcut
oflm pull qwen3.5:4b
oflm run qwen3.5:4b
```

Pass `--family` and `--xclbin-from` when installing from a local folder or a custom fine-tune whose name hides the base architecture:

```bash
oflm add <repo-or-dir> --family qwen3.6-moe --xclbin-from Qwen3.6-35B-A3B-NPU2
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Context & RAM | Status |
|---|---|---|---|---|
| Custom Qwen 3 Text GGUF (`oflm add`) | {{< b "size" "0.6B" >}} {{< b "size" "1.7B" >}} {{< b "size" "4B" >}} {{< b "size" "8B" >}} {{< b "size" "14B" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "128K" >}} {{< b "size" "8–32 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| Custom Qwen 2.5 Text GGUF (`oflm add`) | {{< b "size" "3B (Qwen 2.5)" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "32K" >}} {{< b "size" "8–16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| Custom Qwen 3-VL / Qwen 2.5-VL GGUF (`oflm add`) | {{< b "size" "3B (Qwen 2.5-VL)" >}} {{< b "size" "VL 4B" >}} {{< b "size" "VL 8B" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Dense language layers run on the NPU; ViT image encoding runs on the CPU via mmproj*.gguf." >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "32K–128K" >}} {{< b "size" "16–32 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| Custom Qwen 3.5 GGUF (`oflm add`) | {{< b "size" "0.8B" >}} {{< b "size" "2B" >}} {{< b "size" "4B" >}} {{< b "size" "9B" >}} {{< b "size" "27B" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Linear & gated attention layers and FFN projections run on the NPU; recurrent state updates during multi-token prompt prefill and ViT image encoding (when images are attached) run on the CPU." >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "256K" >}} {{< b "size" "8–32 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| Custom Qwen 3.6 MoE GGUF (`oflm add`) | {{< b "size" "35B-A3B" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Shared layers, routed expert projections, and attention run on the NPU; top-k expert routing, long-prompt recurrent state updates, and ViT image encoding (when images are attached) run on the CPU." >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "256K" >}} {{< b "size" "64 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `qwen3:0.6b`, `qwen3:1.7b` | {{< b "size" "0.6B" >}} {{< b "size" "1.7B" >}} | {{< b "open" "NPU" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K / 128K" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `qwen3:4b`, `qwen3-it:4b`, `qwen3-tk:4b` | {{< b "size" "4B" >}} | {{< b "open" "NPU" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K / 128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `qwen3:8b` | {{< b "size" "8B" >}} | {{< b "open" "NPU" >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "16K / 128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `qwen3vl-it:4b` | {{< b "size" "VL 4B" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Language layers run on the NPU; ViT image encoding runs on the CPU when images are attached." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K / 128K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `qwen2.5vl-it:3b` | {{< b "size" "3B (Qwen 2.5-VL)" >}} | {{< b k="closed" t="Closed NPU" tip="Lacks open_kernels symlink to Qwen2.5-3B-Instruct-NPU2; requires OFLM_QWEN_ENGINE=closed (omitted on source-built NixOS #244). Install a Qwen 2.5-VL 3B GGUF via oflm add for open NPU + CPU execution." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "wip" "244" "NixOS #244" >}} |
| `qwen3.5:0.8b`, `qwen3.5:2b` | {{< b "size" "0.8B" >}} {{< b "size" "2B" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Linear & gated attention layers and FFN projections run on the NPU; recurrent state updates during multi-token prompt prefill and ViT image encoding (when images are attached) run on the CPU." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K / 256K" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `qwen3.5:4b`, `qwen3.5:9b` | {{< b "size" "4B" >}} {{< b "size" "9B" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Linear & gated attention layers and FFN projections run on the NPU; recurrent state updates during multi-token prompt prefill and ViT image encoding (when images are attached) run on the CPU." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K / 256K" >}} {{< b "size" "16 GB+ RAM" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "240" "Shortcut #240" >}} |
| `qwen3.8:27b` | {{< b "size" "27B" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Linear & gated attention layers and FFN projections run on the NPU; recurrent state updates during multi-token prompt prefill run on the CPU (qwen3.8:27b shortcut is text-only)." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K / 256K" >}} {{< b "size" "32 GB+ RAM" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "240" "Shortcut #240" >}} |
| `qwen3.6-moe:35b-a3b` | {{< b "size" "35B-A3B" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Shared layers, routed expert projections, and attention run on the NPU; top-k expert routing, long-prompt recurrent state updates, and ViT image encoding (when images are attached) run on the CPU." >}} {{< b "q4nx" "Legacy Q4NX" >}} | {{< b "ctx" "32K / 256K" >}} {{< b "size" "64 GB+ RAM" >}} | {{< b "ok" "Ready" >}} {{< b "wip" "240" "Shortcut #240" >}} |

## Tips & Current Limitations

{{< wip issue="240" title="GGUF Registry Updates & Qwen3.5-Omni" >}}Issue #240 tracks GGUF registry updates and speculative drafters for `qwen3.5:4b/9b`, `qwen3.8:27b`, `qwen3.6-moe:35b-a3b`, and Ternary Bonsai 2 27B (`PQ2_0`). Issue [#73](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/73) tracks Qwen3.5-Omni audio/multimodal support.{{< /wip >}}

- **Qwen 3 vs. Qwen 2.5 sizes:** Qwen 3 supports `0.6B`, `1.7B`, `4B`, `8B`, `14B`, `VL 4B`, and `VL 8B` (`30B-A3B`, `32B`, and `235B-A22B` are not supported). Qwen 2.5 supports **`3B` only** (`qwen25-3b`, text and VL GGUF; `0.5B`, `1.5B`, `7B`, `14B`, `32B`, and `72B` Qwen 2.5 sizes are not supported). Note that the legacy `qwen2.5vl-it:3b` `.q4nx` shortcut lacks an `open_kernels` symlink to `Qwen2.5-3B-Instruct-NPU2` and requires `OFLM_QWEN_ENGINE=closed` (omitted on NixOS `#244`); install a Qwen 2.5 or 2.5-VL `3B` GGUF via `oflm add` to run on the open runtime.
- **Open runtime vision (`mmproj*.gguf` + `-r`):** Image Q&A works today on the open runtime across Qwen 3-VL (`4B`, `8B`), Qwen 2.5-VL (`3B` GGUF), Qwen 3.5 (`0.8B`, `2B`, `4B`, `9B`, plus `27B` GGUFs with `mmproj`; note the built-in `qwen3.8:27b` `.q4nx` shortcut is text-only while `oflm add unsloth/Qwen3.5-27B-GGUF` uses the same open kernel set), and Qwen 3.6 MoE (`35B-A3B`). `oflm add` automatically downloads `mmproj*.gguf` alongside the model: the ViT image encoder runs on the host CPU while language and MoE layers run on the Open NPU (video inputs are not supported). Control image resolution with `-r` (`0` = original, `1` = 480p, `2` = 720p default, `3` = 1080p, `4` = 1440p):

```bash
oflm serve qwen3vl-it:4b -r 1
```

- **Quantization & memory sizing:** `Q4_1` checkpoints load with zero conversion overhead; OpenFlowLM converts `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `IQ4` GGUF files on first load. Use 32 GB+ of system RAM for `27B` (~18.5 GB on disk) and 64 GB+ for `qwen3.6-moe:35b-a3b` (~22.1 GB on disk) or community 40B layer-scaled merges built on 27B dimensions.
- **Context & tool calling:** Override the default context window with `--ctx-len` (up to `256K` on Qwen 3.5 and Qwen 3.6 MoE; `128K` on Qwen 3; `32K` on Qwen 2.5 `3B`). Qwen 3, Qwen 3.5, and Qwen 3.6 MoE emit `<think>` reasoning traces and handle tool calls in [Server Mode](/docs/instructions/server/tool_calling/).
