---
layout: docs
title: Models
nav_order: 3
has_children: true
---

# 🧩 Models Overview

- 📑 **Detailed model cards are available** -- explore the model families below
- 🚀 **More models are on the way!**
- ⚠️ **Not every model runs on open kernels.** Read the
  [support-status matrix](#open-vs-closed-support-status) before you assume a
  model is covered by this repository's AIE designs.

---

## 📚 Model families

- [LLaMA](llama/)
- [DeepSeek](deepseek/)
- [Qwen](qwen/)
- [Gemma](gemma/)
- [MedGemma](medgemma/)
- [TranslateGemma](translategemma/)
- [gpt-oss](gpt-oss/)
- [LiquidAI/LFM](lfm/)
- [Phi](phi/)
- [Nanbeige](nanbeige/)
- [Whisper](whisper/)
- [EmbeddingGemma](embeddinggemma/) and the
  [BERT-family encoders](#-bert-family-npu-encoders)
- [SmolVLA](smolvla/) *(not yet supported)*

---

## 🔓 Open vs. closed support status

OpenFlowLM replaces FastFlowLM's pre-compiled kernels with open AIE designs built
from source in [`open_kernels/`](https://github.com/Atomic-Germ/OpenFlowLM/tree/main/open_kernels)
and [`npu_offload/`](https://github.com/Atomic-Germ/OpenFlowLM/tree/main/npu_offload).
That coverage is **not complete**. Where no open design exists, OFLM dispatches to
the closed engine `.so`/`.dll` that upstream ships. Both paths run on the NPU; only
one of them is built from source here.

| Family | Tags | Text | Vision / Audio | Notes |
|---|---|---|---|---|
| Qwen 3 / 3.5 / 3.6-MoE | `qwen3:*`, `qwen3-it:4b`, `qwen3-tk:4b`, `qwen3.5:*`, `qwen3.6-moe:35b-a3b` | ✅ **open** | ❌ closed | The open engine has **no vision path** -- passing an image selects the closed engine even when the text path runs open. |
| Qwen 2.5 / 2.5-VL | `qwen2.5-it:3b`, `qwen2.5vl-it:3b` | ✅ open | ❌ closed | |
| Qwen 3-VL | `qwen3vl-it:4b` | ✅ open | ❌ closed | |
| LLaMA 3 | `llama3.1:8b`, `llama3.2:1b`, `llama3.2:3b` | ✅ **open** | — | |
| Gemma 3 (text) | `gemma3:1b` | ✅ **open** | — | |
| Gemma 3 (multimodal) | `gemma3:4b` | ✅ open | ❌ closed | Images always need the closed `gemma_npu` DLL. |
| MedGemma, TranslateGemma | `medgemma:4b`, `medgemma1.5:4b`, `translategemma:4b` | ✅ open | ❌ closed | Same `gemma3` family, same image caveat. |
| Phi-4-mini | `phi4-mini-it:4b` | ✅ **open** | — | |
| LFM2 / LFM2.5 | `lfm2:*`, `lfm2-trans:2.6b`, `lfm2.5-it:1.2b`, `lfm2.5-tk:1.2b` | ✅ **open** | — | Includes the WideDeltaNet recurrent-state chain. |
| Nanbeige 4.1 | `nanbeige4.1:3b` | ✅ **open** | — | |
| **Gemma 4** | `gemma4-it:e2b`, `gemma4-it:e4b`, `gemma4-it:12b` | ❌ **closed** | ❌ closed | **No open design exists.** These dispatch entirely to the closed `gemma4e_npu` / `gemma4_12b_npu` engines. |
| **GPT-OSS** | `gpt-oss:20b`, `gpt-oss-sg:20b` | ❌ **closed** | — | No open design: the MoE experts need a clamped SwiGLU plus bias-carrying projections that no current design has room for. |
| **Whisper** | `whisper-v3:turbo` | ❌ **closed** | ❌ closed | The published container ships only `model.q4nx`, so the open Whisper path is never selected for it. |
| DeepSeek-R1 (distills) | `deepseek-r1:8b`, `deepseek-r1-0528:8b` | ✅ open *(via the base recipe)* | — | These are **distills of Qwen3-8B and Llama-3.1-8B**, so they reuse the `qwen3` and `llama3` recipes respectively rather than having one of their own. |
| EmbeddingGemma | `embed-gemma:300m` | ⚙️ CPU by default | — | Ships unquantized; NPU matmul is opt-in, not the default. |
| BERT-family encoders | `bge-*`, `all-minilm:l6-v2`, `nomic-embed-text:v1.5`, `gte-multilingual:base` | ✅ **open** | — | See below. |

**Legend:** ✅ open = built from source in this repo · ❌ closed = upstream
pre-compiled engine · ⚙️ = neither by default

### Why some families are closed

These are gaps in the open designs, not configuration mistakes on your side. See
[`open_kernels/recipes/families.py`](https://github.com/Atomic-Germ/OpenFlowLM/blob/main/open_kernels/recipes/families.py),
which lists the implemented families and keeps a `NOT_IMPLEMENTED` set with the
reason for each exclusion. `AGENTS.md` describes the workflow for adding one.

---

## 🔢 BERT-family NPU encoders

Beyond EmbeddingGemma, OFLM ships **six** BERT-family encoders that run on the
NPU through the `open_npue` backend -- one compiled design set per encoder shape.
Serve one with `--embed 1 --embeddingmodel <tag>`:

| tag | Parameters | Design | Sequence length | Task prompts |
|---|---|---|---|---|
| `bge-base:en-v1.5` | 109M | `BERT-h768-bfp16` | 512 | — |
| `bge-small:en-v1.5` | 33M | `BERT-h384-bf16` | 512 | — |
| `bge-large:en-v1.5` | 335M | `BERT-h1024-bfp16` | 512 | — |
| `all-minilm:l6-v2` | 22M | `BERT-h384-bfp16` | 512 | — |
| `nomic-embed-text:v1.5` | 137M | `BERT-h768-gated-bfp16` | 512 | ✅ required (`query`, `document`, `clustering`, ...) |
| `gte-multilingual:base` | 305M | `BERT-h768-gated-bfp16` | 512 | — |

All six are bf16 safetensors -- **not** quantized, and not Q4NX containers.
Their sequence length is fixed at 512 by the compiled design rather than by the
request.

`nomic-embed-text:v1.5` declares named task prompts, so it **requires**
`--prompt-name` (or the `prompt_name` REST field) and will refuse to run without
one. The others have no task-prompt concept and refuse the flag outright.
`embed-gemma:300m` is the exception to both rules: it declares no names but still
honours tasks through a hardcoded per-task prefix.

```shell
oflm serve llama3.2:1b --embed 1 --embeddingmodel bge-base:en-v1.5
oflm serve llama3.2:1b --embed 1 --embeddingmodel nomic-embed-text:v1.5 --prompt-name query
```

Benchmark them with `oflm bench-embed`; see
[Embedding benchmarks](/docs/benchmarks/embeddings_results/).

---

## 📝 Adding your own model

For a model of a shape that already has open kernels, you do not need to write
any code -- see [`oflm add`](/docs/instructions/cli/#-add-a-converted-model-oflm-add).
For a genuinely new architecture, see
[`src/create_new_model.md`](https://github.com/Atomic-Germ/OpenFlowLM/blob/main/src/create_new_model.md).
