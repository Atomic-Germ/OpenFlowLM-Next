---
title: "User Guide"
description: "Guides for OpenFlowLM CLI usage, persistent configuration, and local HTTP server mode."
icon: "menu_book"
weight: 20
---

Run chat LLMs, Mixture-of-Experts models, vision-language models, Whisper speech-to-text, text embeddings, and FLUX.2 [klein] 4B image generation offline on **Windows & Linux** with AMD Ryzen AI XDNA2 NPUs.

## Core Guides

<div class="oflm-two-col">
  <div class="oflm-card">

**[CLI Usage →](/docs/instructions/cli/)**

Chat in your terminal, install GGUF models with `oflm add`, generate or edit images with `oflm image`, validate your NPU stack, and run `oflm bench` or `oflm bench-embed`.

  </div>
  <div class="oflm-card">

**[Configuration →](/docs/instructions/config/)**

Persist global and per-model overrides in `settings.json` with `oflm config` and inspect `OFLM_*` environment variables.

  </div>
  <div class="oflm-card">

**[Server Mode →](/docs/instructions/server/)**

Serve OpenAI- and Ollama-compatible endpoints with co-resident speech, embeddings, tool calling, Open WebUI, and image generation and editing.

  </div>
</div>

## Server Mode Guides

| Guide | Topic |
|---|---|
| [Server Basics](/docs/instructions/server/basics/) | How local HTTP inference servers connect apps and SDKs to your NPU |
| [REST API Reference](/docs/instructions/server/openapi/) | OpenAI and Ollama endpoints plus Python and JavaScript SDK examples |
| [Image Generation and Editing](/docs/instructions/server/images/) | Text-to-image generation and reference-photo editing with FLUX.2 [klein] 4B |
| [Tool Calling](/docs/instructions/server/tool_calling/) | Structured function calling in Python scripts and Open WebUI |
| [Open WebUI](/docs/instructions/server/webui/) | Desktop and Docker setup, multi-model queuing, web search, and local RAG |