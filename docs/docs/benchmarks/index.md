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
- [Embeddings](embeddings_results/)

---

## 📏 How to read these tables

**Every number is throughput or latency, produced by `oflm bench`.** That tool
emits three series per context length -- TTFT, prefill tok/s and decode tok/s --
and **nothing else**. In particular there is no energy or power-draw
instrumentation, so this section publishes no watts, joules or
energy-efficiency ratios.

| legend | meaning |
|---|---|
| **OOC** | Out Of Context -- this stage is longer than the model's shipped `default_context_length` (see [`model_list.json`](https://github.com/Atomic-Germ/OpenFlowLM/blob/main/src/model_list.json)). Raising the limit is possible, but a stage beyond it is not a supported configuration. |
| **OOM** | Out Of Memory -- the NPU allocation cap on this machine was exceeded. Only part of system DRAM is reachable by the NPU, so this depends on the test machine's DRAM. |

A model's `default_context_length` is a **default, not a hard ceiling** -- the
model's own `config.json` is the real limit. Where a stage is marked OOC, treat
it as untested rather than as a failure.

Not every page has the same shape. Most report decode and prefill speed against
prompt length. **Embeddings** deliberately does not: encoders have no first
token and no prefill/decode split, so it sweeps **batch size** and reports
texts/s, tokens/s and the cost of not batching.

There are no benchmark pages for models that cannot yet be loaded, or that have
not been measured -- notably Whisper V3 Turbo, the DeepSeek-R1 distills, GPT-OSS
Safeguard, and SmolVLA (unsupported).

---

## 🔁 Reproducing a table

The sweep configs live in [`utilities/bench-configs/`](https://github.com/Atomic-Germ/OpenFlowLM/tree/main/utilities/bench-configs)
and the full methodology is documented in
[that directory's README](https://github.com/Atomic-Germ/OpenFlowLM/blob/main/utilities/bench-configs/README.md).

```shell
oflm bench llama3.2:1b
oflm bench llama3.2:1b -i utilities/bench-configs/bench-8k.json
oflm bench-embed bge-base:en-v1.5 --max-batch 128 --bench-iterations 3
```

> ⚠️ **NPU power mode is worth about 2x, and neither engine reports which one it
> ran in.** `utilities/bench-pair` exists for this: it interleaves the
> configurations you are comparing so drift cannot favour one of them, and
> discards the first round. Three identical back-to-back `oflm bench` runs have
> been observed spanning 9.05 to 17.78 tok/s of decode on the same machine.
> Compare models with `bench-pair`, not with numbers from two separate sittings.

A run also writes a CSV to the current directory. The
[Embeddings page](embeddings_results/) is the one page whose cells are
machine-checked against that CSV, by
`utilities/check_embed_bench_docs.py`.
