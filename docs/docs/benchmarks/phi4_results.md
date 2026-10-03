---
layout: docs
title: Phi4
parent: Benchmarks
nav_order: 6
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with OpenFlowLM (OFLM).

> **Note:** 
> - Results are based on OpenFlowLM v0.9.30.
> - Under OFLM's default NPU power mode (Performance)   
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models. 

> ℹ️ **This version predates the rename.** The numbers below were measured
> on FastFlowLM, before the `flm` → `oflm` rename reset the version series to
> `0.1.0`. No OpenFlowLM build has ever reported a `0.9.x` or `1.0.x` version.
> Treat these as FastFlowLM-era measurements; re-run `oflm bench` on a `0.1.0`
> build if you need numbers from this engine.


---

### **Test System 1:** 

AMD Ryzen™ AI 7 350 (Kraken Point) with 32 GB DRAM; performance is comparable to other Kraken Point systems.

<div style="display:flex; flex-wrap:wrap;">
  <img src="/assets/bench/phi4_mini_decoding.png" style="width:15%; min-width:300px; margin:4px;">
  <img src="/assets/bench/phi4_mini_prefill.png" style="width:15%; min-width:300px; margin:4px;">
</div>

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Phi-4-mini-instruct**  | NPU (OFLM)    | 21.8 | 21.2 | 19.9 | 18.1 | 14.9 | 11.2 | 

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **HW**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|
| **Phi-4-mini-instruct**  | NPU (OFLM)    | 643 | 787 | 857 | 809 | 644 | 447 | 
