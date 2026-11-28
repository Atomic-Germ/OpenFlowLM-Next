---
title: "EmbeddingGemma"
description: "2,048-token text embeddings with Google EmbeddingGemma 300M on AMD Ryzen AI NPUs."
icon: "data_array"
weight: 80
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "embed" "Embeddings" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b k="hybrid" t="NPU + CPU" tip="Linear and FFN projections run on the NPU (npu_matmul_f32); bidirectional attention and vector pooling across 2,048-token inputs run on the CPU (falls back to CPU on source-built NixOS #244)." >}}{{< b "gguf" "GGUF" >}}{{< b "safetensors" "Safetensors" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "300M" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "2048 tok" >}}{{< b k="qbest" t="Q8_0" mono="true" >}}{{< b k="quant" t="BF16" mono="true" >}}{{< b k="quant" t="F16" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Pull the built-in shortcut (downloads ggml-org/embeddinggemma-300M-GGUF Q8_0)
oflm pull embed-gemma:300m

# Or register a local GGUF / Safetensors build under the required embed-gemma:300m tag
oflm add ./my-embeddinggemma --name embed-gemma:300m

# Serve POST /v1/embeddings alongside a chat model (default when --embed 1 is set)
oflm serve gemma3:4b --embed 1 --embeddingmodel embed-gemma:300m
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")
resp = client.embeddings.create(
    model="embed-gemma:300m",
    input="OpenFlowLM runs 2,048-token embeddings on AMD Ryzen AI NPUs.",
)
print(len(resp.data[0].embedding))
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Sequence & RAM | Status |
|---|---|---|---|---|
| Custom GGUF / Safetensors (`oflm add --name embed-gemma:300m`) | {{< b "size" "300M" >}} {{< b "size" "768d" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Linear and FFN projections run on the NPU (npu_matmul_f32); bidirectional attention and vector pooling across 2,048-token inputs run on the CPU (falls back to CPU on source-built NixOS #244)." >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} | {{< b "ctx" "2048 tok" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `embed-gemma:300m` | {{< b "size" "300M" >}} {{< b "size" "768d" >}} | {{< b k="hybrid" t="NPU + CPU" tip="Linear and FFN projections run on the NPU (npu_matmul_f32); bidirectional attention and vector pooling across 2,048-token inputs run on the CPU (falls back to CPU on source-built NixOS #244)." >}} {{< b "gguf" "GGUF" >}} | {{< b "ctx" "2048 tok" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |

## Tips & Current Limitations

- **Built-in GGUF shortcut & `--name embed-gemma:300m`:** `oflm pull embed-gemma:300m` already downloads `ggml-org/embeddinggemma-300M-GGUF` (`Q8_0`) directly. Because the embedding engine dispatches on the exact tag `embed-gemma:300m`, always pass `--name embed-gemma:300m` when registering a custom `BF16`, `F16`, or local `Q8_0` checkpoint via `oflm add`.
- **NixOS CPU fallback (`#244`):** EmbeddingGemma accelerates projections with prebuilt `npu_matmul_f32` NPU kernels, which are omitted from source-built NixOS packages ([#244](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244)); on NixOS, `embed-gemma:300m` falls back to CPU execution.
- **Single-query execution vs. bulk batching:** EmbeddingGemma encodes one text sequence per pass, suiting interactive RAG queries and `2,048`-token chunks. For bulk indexing of chunks under `512` tokens, use [BERT Encoders](/docs/models/bert/) for multi-item batching.
- **Task hints & Open WebUI setup:** Pass `prompt_name` (such as `query` or `document`) in `POST /v1/embeddings` requests or `--prompt-name` at server launch. In Open WebUI ([Server Mode](/docs/instructions/server/webui/)), open **Admin Settings -> Documents**, set **Embedding Model Engine** to **OpenAI**, URL `http://127.0.0.1:52625/v1`, API key `oflm`, and model `embed-gemma:300m`.
