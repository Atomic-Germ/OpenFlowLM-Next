---
layout: docs
title: Benchmarks
nav_order: 4
has_children: true
---

# 📊 Benchmarks Overview

Browse detailed NPU benchmark results for each major model family supported by OpenFlowLM:

- [LLaMA3](llama3_results/)
- [Gemma3](gemma3_results/)
- [Gemma4](gemma4_results/)
- [Qwen2.5](qwen2.5_results/)
- [Qwen3](qwen3_results/)
- [Qwen3.5](qwen3.5_results/)
- [Qwen3.6](qwen3.6_results/)
- [gpt-oss](gpt-oss_results/)
- [LiquidAI/LFM2](lfm2_results/)
- [Microsoft/Phi4](phi4_results/)
- [Nanbeige4.1](nanbeige4.1_results/)
- [SmolVLA](smolvla_results/)
- [Embeddings](embeddings_results/)

Each page includes decoding and prefill speed metrics (tokens per second) and notes about the test setup and hardware.

The **Embeddings** page is the exception, and deliberately so: encoders have no
first token and no prefill/decode split, so it sweeps BATCH SIZE instead of
context length and reports texts/s, latency and the cost of not batching. 