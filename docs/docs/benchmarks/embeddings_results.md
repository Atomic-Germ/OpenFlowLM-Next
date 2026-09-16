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
is what a caller doing one request per text gets. `Speedup` is the ratio.

---

### **Test System:**

AMD Ryzen™ AI 9 HX 370 (Strix Point) with Radeon 890M, 32 GB DRAM. Mains power,
NPU otherwise idle, models run one at a time in one session.

| | |
|---|---|
| command | `oflm bench-embed <tag> --max-batch 128 --bench-iterations 3` |
| measurement | wall clock, end to end — tokenizer + host + array + pooling |
| statistic | mean of 3 iterations, one discarded warm-up, `±` is population σ |
| backend | `open_npue` for the six encoders, `open_embedding` for embed-gemma |

**These are throughput and latency figures for the whole pipeline, and NOT NPU
kernel claims.** The array is shared, so a wall-clock reading measures how busy
the machine was as much as how good the kernels are — see *A note on how easy
this is to get wrong* at the end of this page, which is not a formality.

---

### 🚀 Throughput (texts per second, batched path)

| **Model** | **dims** | **1** | **4** | **8** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 384 | 108.0 | 433.8 | 785.5 | 850.2 | 1035.7 | 1305.0 | 1102.1 |
| **bge-small-en-v1.5** | 384 | 53.3 | 212.5 | 379.4 | 357.0 | 395.8 | 565.4 | 409.7 |
| **bge-base-en-v1.5** | 768 | 39.5 | 154.7 | 255.0 | 228.3 | 240.6 | 311.3 | 253.0 |
| **nomic-embed-text-v1.5** | 768 | 29.6 | 131.5 | 188.9 | 178.5 | 175.1 | 239.0 | 187.4 |
| **gte-multilingual-base** | 768 | 28.0 | 105.1 | 167.0 | 143.7 | 148.9 | 231.1 | 156.8 |
| **bge-large-en-v1.5** | 1024 | 13.5 | 56.0 | 81.8 | 70.0 | 71.6 | 89.9 | 77.6 |
| *EmbeddingGemma-300M* | *768* | *1.4* | *1.6* | *1.6* | *1.6* | — | — | — |

### 🚀 Latency (seconds for the whole call, batched path)

| **Model** | **1** | **4** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 0.0093 | 0.0092 | 0.0188 | 0.0309 | 0.0491 | 0.1162 |
| **bge-small-en-v1.5** | 0.0188 | 0.0188 | 0.0448 | 0.0809 | 0.1140 | 0.3134 |
| **bge-base-en-v1.5** | 0.0253 | 0.0259 | 0.0701 | 0.1330 | 0.2056 | 0.5059 |
| **nomic-embed-text-v1.5** | 0.0338 | 0.0306 | 0.0905 | 0.1864 | 0.2682 | 0.6849 |
| **gte-multilingual-base** | 0.0357 | 0.0381 | 0.1113 | 0.2149 | 0.2769 | 0.8161 |
| **bge-large-en-v1.5** | 0.0741 | 0.0715 | 0.2287 | 0.4468 | 0.7121 | 1.6505 |
| *EmbeddingGemma-300M* | *0.6992* | *2.5648* | *9.9378* | — | — | — |

### 🚀 Batch speedup (looped ÷ batched, same texts)

| **Model** | **1** | **2** | **4** | **8** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 0.99 | 1.84 | 3.49 | 5.99 | 6.64 | 8.07 | 10.05 | 8.53 |
| **bge-small-en-v1.5** | 1.04 | 1.97 | 3.86 | 6.63 | 6.43 | 7.00 | 10.03 | 7.40 |
| **bge-base-en-v1.5** | 0.99 | 1.90 | 3.91 | 6.43 | 5.44 | 5.73 | 7.55 | 6.14 |
| **nomic-embed-text-v1.5** | 1.02 | 1.95 | 4.12 | 5.59 | 5.37 | 5.01 | 6.94 | 5.64 |
| **gte-multilingual-base** | 1.01 | 1.89 | 3.71 | 5.46 | 4.97 | 5.12 | 7.99 | 5.34 |
| **bge-large-en-v1.5** | 0.98 | 1.96 | 3.95 | 5.79 | 4.94 | 5.01 | 6.35 | 5.47 |
| *EmbeddingGemma-300M (control)* | *0.98* | *0.96* | *1.04* | *0.97* | *1.07* | — | — | — |

### 🚀 Token throughput (tokens per second, batched path)

| **Model** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 18758 | 22850 | 28791 | 24314 |
| **bge-small-en-v1.5** | 7876 | 8732 | 12474 | 9039 |
| **bge-base-en-v1.5** | 5037 | 5308 | 6868 | 5582 |
| **nomic-embed-text-v1.5** | 4653 | 4563 | 6228 | 4885 |
| **gte-multilingual-base** | 3880 | 4022 | 6241 | 4235 |
| **bge-large-en-v1.5** | 1543 | 1580 | 1984 | 1711 |
| *EmbeddingGemma-300M* | not reported | — | — | — |

EmbeddingGemma reads `not reported`: `open_embedding` does not return a token
count, and the benchmark prints that rather than a zero — a zero would read as
a real count.

---

### If you want one number

Per model, at its best point and for a single text. **The best point is batch
64 for all six `open_npue` models**. EmbeddingGemma was swept only to 16, and
it has no peak to speak of: it reads 1.6 texts/s at batches 4, 8 AND 16, so
the batch named for it is simply the first that reaches the flat value.

| **Model** | **embeddings/s (peak)** | **at batch** | **one text** |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 1305.0 | 64 | 9.3 ms |
| bge-small-en-v1.5 | 565.4 | 64 | 18.8 ms |
| bge-base-en-v1.5 | 311.3 | 64 | 25.3 ms |
| nomic-embed-text-v1.5 | 239.0 | 64 | 33.8 ms |
| gte-multilingual-base | 231.1 | 64 | 35.7 ms |
| bge-large-en-v1.5 | 89.9 | 64 | 74.1 ms |
| *EmbeddingGemma-300M* | *1.6* | *4* | *699.2 ms* |

Two qualifiers belong with that number, or it misleads:

- **It is a curve, not a point.** One text at a time gives 108.0/s on MiniLM
  and 39.5/s on bge-base — an eighth of the peak or less. Which of the two a caller
  gets is the single largest factor on this page.
- **The texts are ~64 tokens.** The design pads every row to its compiled
  `seq`, which is 64 here, so "311.3 embeddings/s" means 311.3 *short* texts.
  The tokens/s column is the one that survives a different text length.

---

### The control, and why it is the most important row

`EmbeddingGemma-300M` is italicised because it is not competing. It runs on
`open_embedding`, which does **not** override `embed_batch()`, so its two paths
are *literally the same loop*. Its speedup therefore has to read ~1.00× — and it
does, **0.96–1.07× across five batch sizes**.

That is what makes the other six rows mean something. Without it, a 5–10×
column could be an artifact of how the harness times the two paths. With it,
the column is measuring the override and nothing else.

It also sets the scale for the rest of the page: **one text costs
EmbeddingGemma 0.70 s and bge-base 0.025 s**, a ~28× gap. Read that as engine
*and* model, not engine alone — EmbeddingGemma-300M has 2.75× the parameters of
bge-base and a 262k-token vocabulary against 30.5k.

---

### The vectors are identical either way, and that is the point

All seven models printed **`identity gate: all N vectors of the largest batch
are BIT-IDENTICAL to the same text embedded alone`** — 128 vectors for the six
`open_npue` models, 16 for EmbeddingGemma, max absolute difference exactly
`0.000e+00`.

Batching is a scheduling choice, not an arithmetic one. So **no accuracy gate,
no cosine and no bit-identity check can tell the slow path from the fast one**;
the only symptom is time, and until this command existed nothing measured time.
That is why the 5–10× sat in a README rather than in a table, and why the
benchmark compares vectors even though it is a stopwatch: a stopwatch would not
have noticed if the fast path were wrong.

The gate compares **every** row, not just the first. An earlier version checked
only row 0, which is a probe whose coverage nothing verified — an ordering or
truncation error in any later row would still have printed `BIT-IDENTICAL`.

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

The flat stretch across batch **1, 2 and 4** matches the smallest tier: bge-base
costs 0.0253 / 0.0258 / 0.0259 s for one, two and four texts. **A single text
pays for four.** That is the most actionable line on this page for anyone
calling `/v1/embeddings`.

**Beyond that, the tier list alone does not account for the shape**, and the
page says so rather than rounding it away:

- Batch 8 and batch 16 both fit tier 16 and do **not** cost the same
  (0.0314 s against 0.0701 s on bge-base).
- **Batch 64 is the best per-text point on all six `open_npue` models**, and
  128 is worse — 311.3 texts/s against 253.0 on bge-base, and the same
  direction on every one of the six, **reproduced in independent sweeps on
  two days**. Six models agreeing every time is not noise.

What produces that is unresolved. The adapter loads these models with two
concurrent encode lanes, so a plausible explanation is how a request is split
across lanes and tiers rather than the tier list itself — but that is a
hypothesis, not a measurement, and nothing here has tested it.

---

### A note on how easy this is to get wrong

The first attempt at this page was measured in a sweep whose **first** model
started seconds after a link step and a process kill. That model came out
**3.5× slow with fifteen times the spread** (MiniLM: 0.0306 s against 0.0088 s
at batch 1, ±134 against ±8 texts/s), while every later model in the same sweep
landed within 1.4% of an earlier run. Re-running it alone on a quiet machine
reproduced 0.0087 s.

Nothing in the output said anything was wrong. The table was internally
consistent and the identity gate was green, because the vectors were fine — it
was only the *time* that was contaminated, and time is the whole measurement.

So these numbers come from one sweep, in one session, after a settle delay, with
nothing else running, and the contaminated attempt was thrown away rather than
patched. `oflm bench-embed` has **no contention guard** — quiesce the machine
yourself, and do not start a sweep on the heels of a build.

---

### Reproducing

```powershell
oflm bench-embed bge-base:en-v1.5 --max-batch 128 --bench-iterations 3
oflm bench-embed nomic-embed-text:v1.5 --max-batch 128 --bench-iterations 3 --prompt-name query
```

`nomic` needs `--prompt-name` because it declares task prompts and
`/v1/embeddings` requires one; benchmarking it without one would time a request
no client can send, so the command refuses. The five BERT-family models (the
three bge sizes, MiniLM and gte) have no task-prompt concept and refuse the
flag. `embed-gemma:300m` is the odd one out: it declares no prompt names and
still prefixes per task, so it accepts the flag and needs none.

Every run writes `bench_embed_<tag>_<YYYYMMDD>[_<cpu>].csv` into the working
directory, with min/max/σ per stage and a `#` provenance header naming the
model, the task, the corpus and the identity-gate result. The tables above are
the mean columns, generated from the run's own output rather than transcribed —
`utilities/check_embed_bench_docs.py` re-checks every cell on this page against
it.

A shorter sweep is `-i utilities/bench-configs/bench-embed-32.json`.

Full flag documentation, the config-file keys, what the command refuses and why,
and what it deliberately does **not** measure (no NPU-side breakdown, no
contention guard, no energy, no CPU arm) are in
[`utilities/bench-configs/README.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/utilities/bench-configs/README.md).
