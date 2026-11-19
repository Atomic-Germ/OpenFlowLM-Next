---
title: "BERT Encoders"
description: "Batched text embeddings across 6 compiled BGE, MiniLM, NomicBERT, and GTE Multilingual encoders on AMD Ryzen AI NPUs."
icon: "scatter_plot"
weight: 70
---

<div class="oflm-spec-grid">
  <div class="oflm-spec-item"><div class="oflm-spec-key">Capabilities</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "embed" "Embeddings" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Runtime &amp; Format</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "open" "NPU" >}}{{< b "gguf" "GGUF" >}}{{< b "safetensors" "Safetensors" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Variants</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "size" "22M" >}}{{< b "size" "33M" >}}{{< b "size" "109M" >}}{{< b "size" "137M" >}}{{< b "size" "305M" >}}{{< b "size" "335M" >}}{{< b k="unsupported" t="BGE-M3 (568M)" mono="true" >}}{{< b k="unsupported" t="Qwen2-Embed (1.5B/7B)" mono="true" >}}</div></div></div>
  <div class="oflm-spec-item"><div class="oflm-spec-key">Context &amp; Quantization</div><div class="oflm-spec-val"><div class="oflm-pill-row">{{< b "ctx" "512 tok" >}}{{< b k="qbest" t="Q8_0" mono="true" >}}{{< b k="quant" t="BF16" mono="true" >}}{{< b k="quant" t="F16" mono="true" >}}{{< b k="quant" t="F32" mono="true" >}}{{< b k="quant" t="Q6_K" mono="true" >}}{{< b k="quant" t="Q5_K" mono="true" >}}{{< b k="quant" t="Q4_K" mono="true" >}}{{< b k="quant" t="Q4_0" mono="true" >}}</div></div></div>
</div>

## Quick Start

```bash
# Pull any of the 6 built-in encoder tags (fetches upstream BF16 Safetensors)
oflm pull bge-base:en-v1.5

# Or install a GGUF / Safetensors build of one of the 6 encoders (--name <tag>)
oflm add nomic-ai/nomic-embed-text-v1.5-GGUF --name nomic-embed-text:v1.5

# Serve POST /v1/embeddings alongside a chat model
oflm serve llama3.2:1b --embed 1 --embeddingmodel bge-base:en-v1.5
oflm serve llama3.2:1b --embed 1 --embeddingmodel nomic-embed-text:v1.5 --prompt-name query

# Benchmark single and batched NPU throughput
oflm bench-embed bge-base:en-v1.5 --max-batch 128 --bench-iterations 3
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")
resp = client.embeddings.create(
    model="bge-base:en-v1.5",
    input=["First document chunk.", "Second document chunk."],
)
print(len(resp.data), len(resp.data[0].embedding))
```

## Variants & Built-in Shortcuts

| Tag | Variant | Runtime & Format | Sequence & RAM | Status |
|---|---|---|---|---|
| Custom GGUF / Safetensors (`oflm add --name <tag>`) | {{< b "size" "22M–335M" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} | {{< b "ctx" "512 tok" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `all-minilm:l6-v2` | {{< b "size" "22M" >}} {{< b "size" "384d" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} | {{< b "ctx" "512 tok" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `bge-small:en-v1.5` | {{< b "size" "33M" >}} {{< b "size" "384d" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} | {{< b "ctx" "512 tok" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `bge-base:en-v1.5` | {{< b "size" "109M" >}} {{< b "size" "768d" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} | {{< b "ctx" "512 tok" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `nomic-embed-text:v1.5` | {{< b "size" "137M" >}} {{< b "size" "768d" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} | {{< b "ctx" "512 tok" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `gte-multilingual:base` | {{< b "size" "305M" >}} {{< b "size" "768d" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} | {{< b "ctx" "512 tok" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |
| `bge-large:en-v1.5` | {{< b "size" "335M" >}} {{< b "size" "1024d" >}} | {{< b "open" "NPU" >}} {{< b "gguf" "GGUF" >}} {{< b "safetensors" "Safetensors" >}} | {{< b "ctx" "512 tok" >}} {{< b "size" "8 GB+ RAM" >}} | {{< b "ok" "Ready" >}} |

## Tips & Current Limitations

- **Supported encoder tags:** The BERT embedding runtime dispatches to the 6 compiled encoder tags in the table above (`all-minilm:l6-v2`, `bge-small:en-v1.5`, `bge-base:en-v1.5`, `nomic-embed-text:v1.5`, `gte-multilingual:base`, and `bge-large:en-v1.5`), not arbitrary uncompiled BERT architectures. `oflm pull <tag>` fetches upstream `BF16` Safetensors directly; when installing a custom GGUF or local Safetensors build with `oflm add`, pass `--name <tag>` so it registers under the matching encoder tag.
- **Quantization & loading:** `Q8_0` GGUF and upstream `Safetensors` checkpoints load with minimal startup overhead; OpenFlowLM converts `BF16`, `F16`, `F32`, `Q6_K`, `Q5_K`, `Q4_K`, and `Q4_0` weights on first load.
- **NixOS package (`.#oflm-with-bert`):** Source-built `.#oflm` on NixOS omits git-tracked `src/xclbins` ([#244](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244)); build or install `.#oflm-with-bert` on NixOS to include the compiled open BERT NPU kernels.
- **Batched indexing & 512-token limit:** Pass up to 128 strings per `input` array for single-pass batched NPU execution. `oflm serve` returns HTTP `400` for inputs over `512` tokens; use [EmbeddingGemma](/docs/models/embeddinggemma/) for passages up to `2,048` tokens.
- **Nomic task prefixes (`--prompt-name`):** `nomic-embed-text:v1.5` expects a task prefix (`query`, `document`, `clustering`, or `classification`) via `--prompt-name` at server startup or `prompt_name` in the JSON request body.
