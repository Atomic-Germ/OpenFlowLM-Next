---
layout: docs
title: Embeddings
parent: Benchmarks
nav_order: 13
---

## ⚡ Embedding Benchmarks

This page reports the performance of the **embedding** models on NPU with
OpenFlowLM (OFLM). It is produced by `oflm bench-embed`, the encoder sibling of
`oflm bench`.

> **Note:**
> - Results are from OFLM v0.1.0 (dev build), commit `743bc39` plus the
>   `bench-embed` change.
> - Under OFLM's default NPU power mode (Performance).
> - Newer versions may deliver improved performance.
> - Fine-tuned models show performance comparable to their base models: a
>   design set is keyed by **GEMM geometry**, not by fine-tune name.

---

### Why this page is not shaped like the others

Every other benchmark page reports **TTFT**, **prefill tok/s** and **decode
tok/s** across context lengths. Not one of those three exists for an encoder.
There is no first token, no prefill/decode split, and the sequence length is
fixed by the compiled design rather than by the request.

The axis that costs for an encoder is the **batch**, so that is what is swept:
1, 2, 4 … 128 texts per call. And the number that matters most is not on the
other pages at all —

> **How much does a caller lose by sending N requests of one text instead of
> one request of N texts?** On these models, between **5× and 10×**.

Each stage therefore times **two paths over the same texts**: one
`embed_batch()` call, and the same texts one `embed()` call at a time — which
is what `/v1/embeddings` does per input. `Speedup` is the ratio.

---

### **Test System:**

AMD Ryzen™ AI 9 HX 370 (Strix Point) with Radeon 890M, 32 GB DRAM. Mains power,
NPU otherwise idle, models run one at a time.

| | |
|---|---|
| command | `oflm bench-embed <tag> --max-batch 128 --bench-iterations 3` |
| measurement | wall clock, end to end — tokenizer + host + array + pooling |
| statistic | mean of 3 iterations, one discarded warm-up, `±` is population σ |
| backend | `open_npue` for the six encoders, `open_embedding` for embed-gemma |

**These are throughput and latency figures for the whole pipeline, and NOT NPU
kernel claims.** The array is shared, so a wall-clock reading measures how busy
the machine was as much as how good the kernels are.

---

### 🚀 Throughput (texts per second, batched path)

| **Model** | **dims** | **1** | **4** | **8** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 384 | 113.9 | 432.2 | 841.0 | 848.4 | 1040.1 | **1321.8** | 1117.8 |
| **bge-small-en-v1.5** | 384 | 54.4 | 216.2 | 400.2 | 365.9 | 411.5 | **581.8** | 425.5 |
| **bge-base-en-v1.5** | 768 | 40.3 | 157.6 | 277.2 | 241.7 | 231.0 | **309.9** | 250.1 |
| **nomic-embed-text-v1.5** | 768 | 35.6 | 141.0 | 227.0 | 192.7 | 193.4 | **252.5** | 206.8 |
| **gte-multilingual-base** | 768 | 28.7 | 112.5 | 200.1 | 146.9 | 148.9 | **230.6** | 158.3 |
| **bge-large-en-v1.5** | 1024 | 14.5 | 57.1 | 87.0 | 71.0 | 72.2 | **93.0** | 77.8 |
| *EmbeddingGemma-300M* | *768* | *1.6* | *1.9* | *1.8* | — | — | — | — |

### 🚀 Latency (seconds for the whole call, batched path)

| **Model** | **1** | **4** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 0.0088 | 0.0093 | 0.0189 | 0.0308 | 0.0484 | 0.1145 |
| **bge-small-en-v1.5** | 0.0184 | 0.0185 | 0.0438 | 0.0778 | 0.1100 | 0.3009 |
| **bge-base-en-v1.5** | 0.0248 | 0.0254 | 0.0662 | 0.1389 | 0.2065 | 0.5128 |
| **nomic-embed-text-v1.5** | 0.0281 | 0.0284 | 0.0830 | 0.1655 | 0.2535 | 0.6191 |
| **gte-multilingual-base** | 0.0348 | 0.0356 | 0.1089 | 0.2149 | 0.2776 | 0.8088 |
| **bge-large-en-v1.5** | 0.0690 | 0.0701 | 0.2254 | 0.4434 | 0.6880 | 1.6453 |
| *EmbeddingGemma-300M* | *0.6419* | *2.0864* | *8.9930* | — | — | — |

### 🚀 Batch speedup (looped ÷ batched, same texts)

| **Model** | **1** | **2** | **4** | **8** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 0.87 | 1.76 | 3.28 | 6.42 | 6.42 | 7.88 | **10.00** | 8.47 |
| **bge-small-en-v1.5** | 0.94 | 1.87 | 3.79 | 7.03 | 6.50 | 7.18 | **10.27** | 7.43 |
| **bge-base-en-v1.5** | 0.96 | 1.87 | 3.70 | 6.56 | 5.68 | 5.42 | **7.27** | 5.86 |
| **nomic-embed-text-v1.5** | 0.97 | 1.90 | 3.81 | 6.14 | 5.20 | 5.31 | **6.85** | 5.62 |
| **gte-multilingual-base** | 0.98 | 1.91 | 3.72 | 6.65 | 4.89 | 4.97 | **7.72** | 5.31 |
| **bge-large-en-v1.5** | 0.97 | 1.96 | 3.87 | 5.96 | 4.81 | 4.90 | **6.31** | 5.30 |
| *EmbeddingGemma-300M (control)* | *0.96* | *1.03* | *1.08* | *0.99* | *0.96* | — | — | — |

### 🚀 Token throughput (tokens per second, batched path)

| **Model** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 18717 | 22947 | **29162** | 24661 |
| **bge-small-en-v1.5** | 8072 | 9080 | **12836** | 9388 |
| **bge-base-en-v1.5** | 5333 | 5097 | **6838** | 5518 |
| **nomic-embed-text-v1.5** | 5023 | 5041 | **6580** | 5389 |
| **gte-multilingual-base** | 3966 | 4021 | **6226** | 4274 |
| **bge-large-en-v1.5** | 1567 | 1592 | **2053** | 1716 |

EmbeddingGemma reads `not reported`: `open_embedding` does not return a token
count, and the benchmark prints that rather than a zero — a zero would read as
a real count.

---

### The control, and why it is the most important row

`EmbeddingGemma-300M` is italicised because it is not competing. It runs on
`open_embedding`, which does **not** override `embed_batch()`, so its two paths
are *literally the same loop*. Its speedup therefore has to read ~1.00× — and it
does, **0.96–1.08× across five batch sizes**.

That is what makes the other six rows mean something. Without it, a 5–10×
column could be an artifact of how the harness times the two paths. With it,
the column is measuring the override and nothing else.

It also sets the scale for the rest of the page: **one text costs
EmbeddingGemma 0.64 s and bge-base 0.025 s**, a ~26× gap. Read that as engine
*and* model, not engine alone — EmbeddingGemma-300M has 2.75× the parameters of
bge-base and a 262k-token vocabulary against 30.5k.

---

### The vectors are identical either way, and that is the point

All seven models printed **`batched and looped paths returned BIT-IDENTICAL
vectors`** — max absolute difference exactly `0.000e+00` on the compared vector.

Batching is a scheduling choice, not an arithmetic one. So **no accuracy gate,
no cosine and no bit-identity check can tell the slow path from the fast one**;
the only symptom is time, and until this command existed nothing measured time.
That is why the 5–10× sat in a README rather than in a table, and why the
benchmark compares a vector from each path even though it is a stopwatch: a
stopwatch would not have noticed if the fast path were wrong.

---

### Reading the curve: batch tiers, and one thing they do not explain

A design is compiled with a set of **batch tiers** and a request is right-sized
to one. You do not have to infer them — the engine prints them in its own
banner above the table:

```
  tiers      4, 16, 32, 128  (requests are right-sized, not padded)
  shape      batch 128 x seq 64  (M = 8192)
  datapath   bfp16-emulated MMAC, C as bf16
```

The flat stretch across batch **1, 2 and 4** matches the smallest tier exactly,
on every model: bge-base costs 0.0248 / 0.0255 / 0.0254 s for one, two and four
texts. **A single text pays for four.** That is the single most actionable line
on this page for anyone calling `/v1/embeddings`.

**Beyond that, the tier list alone does not account for the shape**, and the
page says so rather than rounding it away:

- Batch 8 and batch 16 both fit tier 16 and do **not** cost the same
  (0.0289 s against 0.0662 s on bge-base).
- **Batch 64 is the best per-text point on all six `open_npue` models**, and
  128 is worse -- 309.9 texts/s against 250.1 on bge-base, and the same direction on
  every one of the six. Consistent across six models is not noise.

What produces that is unresolved. The adapter loads these models with two
concurrent encode lanes, so a plausible explanation is how a request is split
across lanes and tiers rather than the tier list itself — but that is a
hypothesis, not a measurement, and nothing here has tested it.

---

### Reproducing

```powershell
oflm bench-embed bge-base:en-v1.5 --max-batch 128 --bench-iterations 3
```

Every run writes `bench_embed_<tag>_<YYYYMMDD>[_<cpu>].csv` into the working
directory, with min/max/σ per stage — the tables above are the mean columns. A
shorter sweep is `-i utilities/bench-configs/bench-embed-32.json`.

Full flag documentation, the config-file keys, and what the command deliberately
does **not** measure (no NPU-side breakdown, no contention guard, no energy, no
CPU arm) are in
[`utilities/bench-configs/README.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/utilities/bench-configs/README.md).
