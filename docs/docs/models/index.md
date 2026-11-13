---
layout: docs
title: Models
nav_order: 3
has_children: true
---

# 🧩 Models Overview

- 📑 **Detailed model cards are available** -- explore the model families below
- 🚀 **More models are on the way!**
- 🔧 **A fine-tune is not a new model.** The kernels follow the model's
  *geometry*; a fine-tune of a base architecture runs on the base's kernel
  set. Read the [geometry vs. fine-tuning note](#geometry-is-the-unit-not-the-finetune)
  below — it shows a 40B model running happily on a 27B set.
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
That coverage is **not complete**. Open is the default and there is **no silent
fallback to the closed engine**: if the open kernels for a model are not installed,
OFLM refuses and tells you to `oflm add` it. The closed `.so`/`.dll` upstream ships is
reachable only by explicitly setting the family's engine switch
(`OFLM_<FAMILY>_ENGINE=closed`, e.g. `OFLM_QWEN3_ENGINE=closed`) -- it exists as a
legacy escape hatch, not as an automatic path, and it is not built from source here.
Note the closed engine reads the upstream `.q4nx` container, so it cannot serve a
GGUF-installed model at all.

In this build, the closed engines for the fully-open families are **removed
entirely**: Llama 3, Qwen 2.5, Qwen 3 (plus the `-it:4b` / `-tk:4b` derivative
tags), LFM2/LFM2.5, Phi-4-mini, Nanbeige, Gemma 3 1B (text) and the DeepSeek-R1
distill tags (which run on the Llama-3.1-8B and Qwen3-8B open bundles). Their
prebuilt kernels and standalone engine test projects are gone with them. What
remains closed is exactly the escape hatches for unserved operations: the VL/omni
engines (images, omni), `gemma_npu` (Gemma 3 vision, used via
`OFLM_GEMMA_ENGINE=closed`), `gemma4_12b_npu` (Gemma 4 12B vision), `gemma4e_npu`
and `gpt_oss_npu` (no open design yet), and `whisper_npu`.

| Family | Tags | Text | Vision / Audio | Notes |
|---|---|---|---|---|
| Qwen 3 / 3.5 / 3.6-MoE | `qwen3:*`, `qwen3-it:4b`, `qwen3-tk:4b`, `qwen3.5:*`, `qwen3.6-moe:35b-a3b` | ✅ **open** | ❌ closed | Text is open (the VL/omni engines can serve VL models via their mmproj weights; otherwise `OFLM_*_ENGINE=closed`). The `qwen3-it:4b` / `qwen3-tk:4b` derivative tags resolve into the `Qwen3-4B-NPU2` open bundle at install time. |
| Qwen 2.5 / 2.5-VL | `qwen2.5-it:3b`, `qwen2.5vl-it:3b` | ✅ open | ❌ closed | |
| Qwen 3-VL | `qwen3vl-it:4b` | ✅ open | ❌ closed | |
| LLaMA 3 | `llama3.1:8b`, `llama3.2:1b`, `llama3.2:3b` | ✅ **open** | — | |
| Gemma 3 | `gemma3:4b`, `gemma3:12b` | ✅ **open** | ❌ closed (`OFLM_GEMMA_ENGINE=closed`) | Text runs on the open `gemma3` kernel set (4B/12B geometries; 1B never got one). Vision/audio needs the closed `gemma_npu` DLL via `OFLM_GEMMA_ENGINE=closed`. |
| Gemma 3 (text) | `gemma3:1b` | ❌ **not available** | — | No open set for the 1B class, and the closed `gemma_text_npu` engine has been removed. Not runnable in this build. |
| MedGemma, TranslateGemma | `medgemma:4b`, `medgemma1.5:4b`, `translategemma:4b` | ✅ **open** (text) | ❌ closed (`OFLM_GEMMA_ENGINE=closed`) | Same open gemma3 4B geometry; vision needs the closed engine. |
| Phi-4-mini | `phi4-mini-it:4b` | ✅ **open** | — | |
| LFM2 / LFM2.5 | `lfm2:*`, `lfm2-trans:2.6b`, `lfm2.5-it:1.2b`, `lfm2.5-tk:1.2b` | ✅ **open** | — | Includes the WideDeltaNet recurrent-state chain. |
| Nanbeige 4.1 | `nanbeige4.1:3b` | ✅ **open** | — | |
| **Gemma 4** | `gemma4-it:12b` | ✅ **open** | — | The 12B runs on the open kernels (two dense geometries: sliding-window layers + global layers with proportional RoPE). |
| **Gemma 4** | `gemma4-it:e2b`, `gemma4-it:e4b` | ✅ **open** with the GGUF (text) | ❌ closed | The tags as published (FLM's `Gemma4-E*B-IT-NPU2` containers, a `model.q4nx`) run on the closed `gemma4e_npu` engine, images and audio included. For text on the open kernels, install Google's QAT GGUF with `oflm-add google/gemma-4-E4B-it-qat-q4_0-gguf` (or `-E2B-`) and run the tag it prints; that install is text only. |
| **GPT-OSS** | `gpt-oss:20b`, `gpt-oss-sg:20b` | ❌ **closed** | — | No open design: the MoE experts need a clamped SwiGLU plus bias-carrying projections that no current design has room for. |
| **Whisper** | `whisper-v3:turbo` | ✅ **open** | — | From this repository, the open engine is the default. `oflm add <whisper repo>` builds the open container from the base checkpoint (or reads a llm.cpp-layout GGUF directly) and registers `whisper-v3:turbo`; the closed container is still available via `OFLM_WHISPER_ENGINE=closed`. See [the Whisper card](/docs/models/whisper/#installing-the-open-whisper-build). |
| **FLUX.2 klein** | `flux2-klein:4b`, any `FLUX.2-klein-4B-*` GGUF | ✅ **open** | — | Black Forest Labs' flow-matching diffusion transformer runs on the open kernels end to end; `oflm add` installs it directly from a GGUF repo. |
| **FLUX.2 klein 9B** | `FLUX.2-klein-9B-*` GGUF | ⚠️ open, broken output | — | Runs on the same open kernels and loads, but the images are garbled today. Not usable. |
| DeepSeek-R1 (distills) | `deepseek-r1:8b`, `deepseek-r1-0528:8b` | ✅ open *(via the base recipe)* | — | These are **distills of Llama-3.1-8B and Qwen3-8B**, so they run the `llama3` / `qwen3` recipe and their open kernel bundles. The derivative tags resolve into the family bundle at install time (`xclbins/Deepseek-R1-Distill-Llama-8B-NPU2` → `Llama-3.1-8B-NPU2`, `DeepSeek-R1-0528-Qwen3-8B-NPU2` → `Qwen3-8B-NPU2`). |
| EmbeddingGemma | `embed-gemma:300m` | ⚙️ CPU by default | — | Ships unquantized; NPU matmul is opt-in, not the default. |
| BERT-family encoders | `bge-*`, `all-minilm:l6-v2`, `nomic-embed-text:v1.5`, `gte-multilingual:base` | ✅ **open** | — | See below. |

**Legend:** ✅ open = built from source in this repo · ❌ closed = upstream
pre-compiled engine · ⚙️ = neither by default

### Geometry is the unit, not the fine-tune

The open kernels are compiled for a **geometry** (hidden width, layer count,
head geometry, vocab, FFN width, the layer types) — not for "model X". So a
repo that is a fine-tune of an architecture with open kernels runs on those
kernels, with no new build, no new container, nothing but the install. And it
does not matter whether the fine-tune's maker had AMD's attention:

```shell
oflm add mradermacher/Qwen3.8-40B-Ultimate-Heretic-Uncensored-Build2-i1-GGUF
# [INFO] Qwen3.8-40B-...-Q4_1.gguf: 1290 tensors (qwen35), quant ['Q4_1', 'Q6_K']
# [INFO] family 'qwen3.5' from the GGUF architecture
# [INFO] xclbins from qwen3.8:27b (no exact size match for 40B; using 27B kernels)
```

That installs and *runs* (`oflm run qwen3.8-ultimate:40b` answers
sensibly; it is not fast, but it is correct), because the geometry it exposes
is the geometry of the Qwen3.8 family and the 27B set serves it. The 40B model
has 96 layers — not the 27B's — but the **depth is the manifest's business,
not the kernels'**: the layers each run the same kernel, and the number of
times the loop is unrolled is a function of the model's own layer count.

A fine-tune of a covered family is therefore additive: **if a recipe exists for
the base's geometry, your fine-tune gets it for free**. A fine-tune of an
uncovered base does not get a recipe; it gets an honest refusal by name.

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

If the model's **family** has open kernels, there is nothing to write:

```shell
oflm add <hf-repo | local-dir | local.gguf>
```

`oflm add` reads the weights (a GGUF header, a `.q4nx` header, or a
`config.json`), derives the geometry, links the family kernel set, and registers
the tag. A fine-tune links its base model's kernels. If the size has never been
built and you are running from a source checkout, the same command builds the
kernel set from the spec; an installed package reports the size as one the
release should ship.

**Fine-tunes work out of the box.** There is no separate "fine-tuned model"
path: the open kernels are compiled for a geometry, not for a model. A
fine-tune of any covered family runs on the base set — even at a different
depth (like the 40B linked against the 27B set; see
[above](#geometry-is-the-unit-not-the-finetune)). The registry entry
is per model only for the tag, not for the kernels.

**Quant selection is automatic, and forgiving.** The installer starts from the
most NPU-friendly candidate the repo carries (`Q4_1` first), verifies the
tensor types against the packers, and *falls through to the next-best file* if
one repo's file hides a tensor type the packers can't ingest (a `.gguf` whose
header claims `Q4_1` but holds e.g. a `Q5_0` embedding row is refused; the
installer tries the `q4_0`/`q4_k_s`/`q6_k`/`q8_0` file instead). If every
compatible file fails, the error names every refusal and tells you how to fix
it. You asked for `:Q5_0`? That means it is refused, because the open packers
do not ingest it — but asking for nothing picks the best thing available.

Supported quants (for the chat kernels): `Q4_1`, `Q4_0`, `Q6_K`, `Q8_0`,
`Q5_K`, `Q4_K`, `IQ4`. Embedding encoders accept a wider set (F16/BF16/F32
included). Conversions of other quants are done on the way in.

**Quants: what they cost and what they don't.** For a text model installed
from a GGUF, quant choice affects only (a) size on disk and (b) the time
`oflm add` and the first load spend in packing — *not* output quality. All
quants land in the same q4_1 pool law, so the same weights produce the same
attention. Load-time difference is real (measured: same model, same answer
text, 56 s on a 6B-ish Q6_K vs 10 s on a Q4_1, at the same resident memory) —
the choice is about install convenience, not the running model. The
`Q4_1` family of quants is preferred because it is what the NPU packers read
most directly; a repo's `Q4_1` that turns out to contain a stray tensor the
packers reject (e.g. an embedding row in `Q5_0`) makes the installer fall
through to the next compatible quant in the repo — nothing is wrong with the
model, only with the file.

The other engines work like this too, with different trade-offs: the
diffusion checkpoint is chosen by resolution and edit profile rather than
quant (f16 and q4_1 are quality-equivalent for the bundle layout), and
the embedding encoders run bf16 directly because they were designed as
reference fidelity, not speed.

The shipped `model_list.json` is a suggestion list for `oflm list` and
`oflm pull`, not a gate — the geometry decides.

For a genuinely new architecture (a new recipe and, if needed, new designs),
see [`src/create_new_model.md`](https://github.com/Atomic-Germ/OpenFlowLM/blob/main/src/create_new_model.md).
