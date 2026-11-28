---
title: "Server Basics"
description: "How local AI servers and the OpenAI HTTP API standard connect client apps to your NPU."
icon: "menu_book"
weight: 10
---

`oflm serve` runs an HTTP server on your Ryzen AI machine (`127.0.0.1:52625` by default). Prompts, files, and outputs stay on your device while desktop apps, editors, and SDKs connect over standard JSON endpoints.

## Why Use a Local HTTP Server?

| Advantage | How It Helps |
|---|---|
| Zero client NPU setup | Apps call standard HTTP endpoints without installing C++ or driver libraries |
| Shared NPU residency | Models load once and serve multiple apps and terminals concurrently |
| Built-in queuing & streaming | Request queuing, SSE streaming, chat templates, and tool parsing run in `oflm serve` |
| Drop-in OpenAI switch | Point `base_url` to `http://127.0.0.1:52625/v1` without rewriting client code |

## Dual API Compatibility

OpenFlowLM implements both major local inference protocols out of the box:

<div class="oflm-two-col">
  <div class="oflm-card">

### OpenAI HTTP API (`/v1/*`)

Works with official Python, JS/TS, Go, .NET, and Java SDKs, [Open WebUI](/docs/instructions/server/webui/), Continue, OpenCode, Cursor, and desktop chat clients:

- `/v1/chat/completions` & `/v1/completions`
- `/v1/embeddings` & `/v1/audio/transcriptions`
- `/v1/images/generations` & `/v1/images/edits`

  </div>
  <div class="oflm-card">

### Ollama HTTP API (`/api/*`)

Connects Ollama clients and local automation libraries directly to `http://127.0.0.1:52625`:

- `/api/chat`, `/api/generate`, `/api/embeddings`
- `/api/tags`, `/api/show`, `/api/ps`, `/api/pull`
- `/api/cancel`, `/api/npu/status`

  </div>
</div>

Continue to [REST API Reference](/docs/instructions/server/openapi/), [Image Generation and Editing](/docs/instructions/server/images/), or [Server Mode](/docs/instructions/server/).