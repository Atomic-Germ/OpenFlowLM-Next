---
title: "Server Mode"
description: "Run OpenFlowLM as an OpenAI- and Ollama-compatible HTTP server with multi-engine NPU residency."
icon: "dns"
weight: 40
---

`oflm serve` runs a local OpenAI- and Ollama-compatible HTTP server on **Windows & Linux**, keeping up to one model per capability slot (chat/vision, embeddings, speech, and image generation) resident on the NPU at once.

| Setting | Default |
|---|---|
| **Listen Address** | `http://127.0.0.1:52625` |
| **OpenAI Base URL** | `http://127.0.0.1:52625/v1` |
| **Ollama Base URL** | `http://127.0.0.1:52625` |
| **Concurrency** | `10` queued NPU requests · `10` sockets |

Start `oflm serve` with any tag from `oflm list` (including GGUF models installed via [`oflm add`](/docs/instructions/cli/#add-a-model-oflm-add)):

```bash
oflm serve llama3.2:1b
```

## Server Mode Guides

<div class="oflm-two-col">
  <div class="oflm-card">

**[Server Basics →](/docs/instructions/server/basics/)**

Learn how `oflm serve` exposes OpenAI (`/v1/*`) and Ollama (`/api/*`) protocols to local apps and SDKs.

  </div>
  <div class="oflm-card">

**[REST API Reference →](/docs/instructions/server/openapi/)**

Browse endpoint tables, sampling extensions, and Python and JavaScript SDK examples for chat, vision, audio, embeddings, and images.

  </div>
  <div class="oflm-card">

**[Image Generation and Editing →](/docs/instructions/server/images/)**

Generate and edit `512x512` and `1024x1024` images on the NPU with FLUX.2 [klein] 4B via `/v1/images/*` and `oflm image`.

  </div>
  <div class="oflm-card">

**[Tool Calling →](/docs/instructions/server/tool_calling/)**

Invoke structured JSON Schema functions from OpenFlowLM chat models in Python scripts and Open WebUI.

  </div>
  <div class="oflm-card">

**[Open WebUI →](/docs/instructions/server/webui/)**

Connect Open WebUI Desktop or Docker for multi-model queuing, Google PSE web search, and local document RAG.

  </div>
</div>

## REST Endpoints & Multi-Engine Residency

`oflm serve` keeps up to one model per capability slot resident at the same time (`[tag]`, `--asr 1`, `--embed 1`, `--imagegen 1`).

| Capability | REST Endpoints | Server Slot & Default |
|---|---|---|
| {{< b "llm" >}} {{< b "vlm" >}} {{< b "tools" >}} | `POST /v1/chat/completions`, `/v1/completions`, `/api/chat`, `/api/generate` | Positional `[tag]` (primary chat, vision, and tool-calling slot) |
| {{< b "embed" >}} | `POST /v1/embeddings`, `/api/embeddings` | `--embed 1` (`-e 1`), `--embeddingmodel` (`embed-gemma:300m`) |
| {{< b "asr" >}} | `POST /v1/audio/transcriptions` | `--asr 1` (`-a 1`), `--asrmodel` (`whisper-v3:turbo`) |
| {{< b "image" >}} | `POST /v1/images/generations`, `/v1/images/edits` | `--imagegen 1`, `--imagemodel` (`flux2-klein:4b`; swaps on demand if `0`) |
| **Server Info** | `GET /v1/models`, `/api/tags`, `/api/ps`, `/api/npu/status`, `/api/version` | Installed models, live residency, and NPU telemetry |

```bash
# Single process with LLM + Embeddings + Whisper ASR
oflm serve lfm2:1.2b -e 1 -a 1

# Single process with LLM + co-resident FLUX.2 [klein] 4B
oflm serve llama3.2:1b --imagegen 1

# Dedicated processes on separate ports
oflm serve -e 1 --port 52625
oflm serve -a 1 --port 52627
oflm serve lfm2:1.2b --port 52628
```

> [!NOTE]
> On NixOS, build `.#oflm-with-bert` for NPU BERT embeddings (`--embeddingmodel bge-base:en-v1.5`); `whisper-v3:turbo` (`-a 1`) is unavailable until open Whisper kernels are exported ({{< b k="wip" issue="244" t="Issue #244" >}}, {{< b k="wip" issue="240" t="Issue #240" >}}; see [Nix & NixOS](/docs/install/linux/nix/)).

## Server Flags

| Flag | Default | Notes |
|---|---|---|
| `--ctx-len`, `-c <N>` | model default | Minimum 512; exact token count used without power-of-two rounding (persist via [`oflm config`](/docs/instructions/config/)) |
| `--prefill-chunk-len <N>` | `4096` | Tokens processed per prefill chunk (minimum 512) |
| `--port`, `-p <N>` | `52625` | Per-process HTTP port (`oflm port` prints resolved port; `OFLM_SERVE_PORT` sets persistent default) |
| `--host <addr>` | `127.0.0.1` | Per-process bind address |
| `--q-len`, `-q <N>` | `10` | Max queued NPU requests |
| `--socket`, `-s <N>` | `10` | Max concurrent socket connections (keep `>= --q-len`) |
| `--cors <0\|1>` | `1` | Enable/disable CORS headers |
| `--quiet` | `false` | Reduce download/hash output (redirect `stdout`/`stderr` to silence HTTP logs) |

```bash
oflm serve llama3.2:1b --ctx-len 8192 --port 8000 --q-len 20 --socket 20
```

> [!WARNING]
> Pass `--cors 0` or place an authenticating reverse proxy in front of `oflm serve` whenever `--host` binds beyond `127.0.0.1`. Update client `base_url` settings whenever you change `--port`.
