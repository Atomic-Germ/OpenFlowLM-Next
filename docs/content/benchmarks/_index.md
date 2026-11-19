---
title: "Benchmarks & Hardware Leaderboard"
description: "Measured AMD Ryzen AI NPU throughput, context scaling, batched embedding tables, and community hardware submissions."
url: "/benchmarks/"
aliases:
  - /benchmarks/community/
  - /docs/benchmarks/
  - /docs/benchmarks/embeddings_results/
  - /docs/benchmarks/qwen3.5_results/
kicker: "Measured Performance"
icon: "speed"
echarts: true
community_bench: true
ctas:
  - text: "Submit Run"
    url: "/benchmarks/submit/"
    icon: "upload_file"
  - text: "Browse Models"
    url: "/models/"
    icon: "view_module"
    outline: true
---

<div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-3">
<div class="d-flex flex-wrap align-items-center gap-2">
<h2 class="h4 fw-bold mb-0">Measured NPU Benchmarks</h2>
{{< b k="wip" issue="240" t="Expanded Family Sweeps #240" >}}
</div>
<div class="oflm-pill-tabs" id="oflm-ref-bench-tabs" role="tablist" aria-label="Benchmark chart views">
<button type="button" class="active" data-ref-tab="all">All Runs</button>
<button type="button" data-ref-tab="llm">Chat Context Scaling (1k-32k)</button>
<button type="button" data-ref-tab="embed">Batched Embeddings (7 Models)</button>
</div>
</div>

<div id="oflm-ref-panel-all">
{{< all-runs-chart >}}
</div>

<div id="oflm-ref-panel-llm" class="d-none">
<div class="oflm-pill-row mb-2">
<span class="oflm-group-label">Configuration:</span>
{{< b k="hybrid" t="NPU + CPU" tip="Token generation and projections run on the NPU; light recurrent state updates during long prompts run on the CPU." >}}
{{< b k="quant" t="Q4_K_M" mono="true" >}}
{{< b k="ctx" t="1k" >}} {{< b k="ctx" t="2k" >}} {{< b k="ctx" t="4k" >}} {{< b k="ctx" t="8k" >}} {{< b k="ctx" t="16k" >}} {{< b k="ctx" t="32k" >}}
<span class="small ms-1" style="color: var(--oflm-muted);">Qwen 3.5 2B · Framework Laptop 13 (AMD Ryzen AI 7 350 w/ Radeon 860M, 32 GB DDR5-5600)</span>
</div>

{{< bench-chart data="qwen35_2b_ryzen_ai_350" >}}

</div>

<div id="oflm-ref-panel-embed" class="d-none">
<div class="oflm-pill-row mb-2">
<span class="oflm-group-label">Acceleration:</span>
{{< b k="open" t="NPU" >}}
{{< b k="hybrid" t="NPU + CPU" tip="BERT, BGE, Nomic, MiniLM, and GTE run 100% on the NPU; EmbeddingGemma runs linear projections on the NPU and attention on the CPU." >}}
<span class="small ms-1" style="color: var(--oflm-muted);">AMD Ryzen AI 9 HX 370 w/ Radeon 890M (32 GB RAM, mains power)</span>
</div>

{{< embed-chart >}}

</div>

<details class="oflm-details">
<summary>View Full Measured Embedding Tables (7 Models, Batch 1 to 128)</summary>

End-to-end wall-clock measurements (mean of 3 runs after 1 warm-up on an idle NPU in `performance` power mode). Batched runs match single-text outputs identically (`0.000e+00` max vector difference).

### Throughput (texts per second, batched path)

| **Model** | **dims** | **1** | **4** | **8** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 384 | 108.0 | 433.8 | 785.5 | 850.2 | 1035.7 | 1305.0 | 1102.1 |
| **bge-small-en-v1.5** | 384 | 53.3 | 212.5 | 379.4 | 357.0 | 395.8 | 565.4 | 409.7 |
| **bge-base-en-v1.5** | 768 | 39.5 | 154.7 | 255.0 | 228.3 | 240.6 | 311.3 | 253.0 |
| **nomic-embed-text-v1.5** | 768 | 29.6 | 131.5 | 188.9 | 178.5 | 175.1 | 239.0 | 187.4 |
| **gte-multilingual-base** | 768 | 28.0 | 105.1 | 167.0 | 143.7 | 148.9 | 231.1 | 156.8 |
| **bge-large-en-v1.5** | 1024 | 13.5 | 56.0 | 81.8 | 70.0 | 71.6 | 89.9 | 77.6 |
| *EmbeddingGemma-300M* | *768* | *1.4* | *1.6* | *1.6* | *1.6* | - | - | - |

### Latency (seconds for the whole call, batched path)

| **Model** | **1** | **4** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 0.0093 | 0.0092 | 0.0188 | 0.0309 | 0.0491 | 0.1162 |
| **bge-small-en-v1.5** | 0.0188 | 0.0188 | 0.0448 | 0.0809 | 0.1140 | 0.3134 |
| **bge-base-en-v1.5** | 0.0253 | 0.0259 | 0.0701 | 0.1330 | 0.2056 | 0.5059 |
| **nomic-embed-text-v1.5** | 0.0338 | 0.0306 | 0.0905 | 0.1864 | 0.2682 | 0.6849 |
| **gte-multilingual-base** | 0.0357 | 0.0381 | 0.1113 | 0.2149 | 0.2769 | 0.8161 |
| **bge-large-en-v1.5** | 0.0741 | 0.0715 | 0.2287 | 0.4468 | 0.7121 | 1.6505 |
| *EmbeddingGemma-300M* | *0.6992* | *2.5648* | *9.9378* | - | - | - |

### Batch speedup (looped ÷ batched, same texts)

| **Model** | **1** | **2** | **4** | **8** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 0.99 | 1.84 | 3.49 | 5.99 | 6.64 | 8.07 | 10.05 | 8.53 |
| **bge-small-en-v1.5** | 1.04 | 1.97 | 3.86 | 6.63 | 6.43 | 7.00 | 10.03 | 7.40 |
| **bge-base-en-v1.5** | 0.99 | 1.90 | 3.91 | 6.43 | 5.44 | 5.73 | 7.55 | 6.14 |
| **nomic-embed-text-v1.5** | 1.02 | 1.95 | 4.12 | 5.59 | 5.37 | 5.01 | 6.94 | 5.64 |
| **gte-multilingual-base** | 1.01 | 1.89 | 3.71 | 5.46 | 4.97 | 5.12 | 7.99 | 5.34 |
| **bge-large-en-v1.5** | 0.98 | 1.96 | 3.95 | 5.79 | 4.94 | 5.01 | 6.35 | 5.47 |
| *EmbeddingGemma-300M (control)* | *0.98* | *0.96* | *1.04* | *0.97* | *1.07* | - | - | - |

### Token throughput (tokens per second, batched path)

| **Model** | **16** | **32** | **64** | **128** |
|---|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | 18758 | 22850 | 28791 | 24314 |
| **bge-small-en-v1.5** | 7876 | 8732 | 12474 | 9039 |
| **bge-base-en-v1.5** | 5037 | 5308 | 6868 | 5582 |
| **nomic-embed-text-v1.5** | 4653 | 4563 | 6228 | 4885 |
| **gte-multilingual-base** | 3880 | 4022 | 6241 | 4235 |
| **bge-large-en-v1.5** | 1543 | 1580 | 1984 | 1711 |
| *EmbeddingGemma-300M* | not reported | - | - | - |

### Summary at peak batch and single text

| **Model** | **embeddings/s (peak)** | **at batch** | **one text** |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 1305.0 | 64 | 9.3 ms |
| bge-small-en-v1.5 | 565.4 | 64 | 18.8 ms |
| bge-base-en-v1.5 | 311.3 | 64 | 25.3 ms |
| nomic-embed-text-v1.5 | 239.0 | 64 | 33.8 ms |
| gte-multilingual-base | 231.1 | 64 | 35.7 ms |
| bge-large-en-v1.5 | 89.9 | 64 | 74.1 ms |
| *EmbeddingGemma-300M* | *1.6* | *4* | *699.2 ms* |

</details>

<script>
(function () {
  const btns = document.querySelectorAll('#oflm-ref-bench-tabs button');
  const panels = {
    all: document.getElementById('oflm-ref-panel-all'),
    llm: document.getElementById('oflm-ref-panel-llm'),
    embed: document.getElementById('oflm-ref-panel-embed')
  };
  if (!btns.length) return;
  btns.forEach(function (btn) {
    btn.addEventListener('click', function () {
      btns.forEach(function (b) { b.classList.remove('active'); });
      btn.classList.add('active');
      const key = btn.dataset.refTab;
      Object.keys(panels).forEach(function (k) {
        if (panels[k]) panels[k].classList.toggle('d-none', k !== key);
      });
      window.dispatchEvent(new Event('resize'));
    });
  });
})();
</script>

<div class="mt-4 pt-2" data-cb-page="leaderboard">

<div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-3">
<div>
<h2 class="h4 fw-bold mb-1">Runs &amp; Community Leaderboard</h2>
<p class="small mb-0" style="color: var(--oflm-muted);">Select any row to inspect full context or batch scaling curves and system specs.</p>
</div>
<a href="submit/" class="btn btn-sm btn-primary"><span class="material-icons align-middle size-18 me-1">upload_file</span>Submit Run</a>
</div>

<div id="cb-seed-banner" class="alert alert-info d-none mb-3" role="status">
<div class="small mb-0"><strong>Verified Runs &amp; Community Submissions:</strong> Static preview shows repository sweeps and averaged community runs; live submissions load automatically on <code>openflowlm.org</code>. Share your hardware speeds with <code>oflm bench &lt;tag&gt; --submit</code> or <a href="submit/">Submit Run</a>.</div>
</div>

<div class="oflm-stat-grid" id="cb-stats">
<div class="oflm-stat-card">
<div class="oflm-stat-label">Displayed Runs</div>
<div class="oflm-stat-value" id="cb-stat-runs">-</div>
<div class="oflm-stat-note" id="cb-stat-runs-note">Validated NPU runs</div>
</div>
<div class="oflm-stat-card">
<div class="oflm-stat-label">Distinct Models</div>
<div class="oflm-stat-value" id="cb-stat-models">-</div>
<div class="oflm-stat-note">Chat &amp; Embedding tags</div>
</div>
<div class="oflm-stat-card">
<div class="oflm-stat-label">Hardware Platforms</div>
<div class="oflm-stat-value" id="cb-stat-hw">-</div>
<div class="oflm-stat-note">Ryzen AI systems</div>
</div>
<div class="oflm-stat-card">
<div class="oflm-stat-label">Top Chat Speed</div>
<div class="oflm-stat-value" id="cb-stat-top-llm">-</div>
<div class="oflm-stat-note" id="cb-stat-top-llm-note">Headline generation</div>
</div>
</div>

<div class="oflm-filter-bar" role="search" aria-label="Filter benchmark runs">
<div>
<label for="f-model">Model Tag</label>
<input type="search" id="f-model" placeholder="e.g. bge-base, qwen3.5..." autocomplete="off" />
</div>
<div>
<label for="f-task">Capability</label>
<select id="f-task">
<option value="">All capabilities</option>
<option value="LLM">Chat &amp; Reasoning</option>
<option value="Embedding">Embeddings (Batch Sweep)</option>
<option value="VLM">Vision</option>
<option value="ASR">Speech-to-Text</option>
<option value="Image">Image Generation</option>
</select>
</div>
<div>
<label for="f-hardware">Laptop / Processor</label>
<input type="search" id="f-hardware" placeholder="e.g. HX 370, Ryzen AI 350..." autocomplete="off" />
</div>
<div>
<label for="f-backend">Runtime</label>
<select id="f-backend">
<option value="">All runtimes</option>
<option value="open_qwen">Chat &amp; Vision</option>
<option value="open_npue">Batched Embeddings</option>
<option value="open_embedding">EmbeddingGemma</option>
</select>
</div>
<div>
<label for="f-quant">Quantization</label>
<input type="search" id="f-quant" placeholder="e.g. Q4_1, Q4_K_M, BF16..." autocomplete="off" />
</div>
<div>
<label for="f-sort">Sort Order</label>
<select id="f-sort">
<option value="decode">Generation / Texts/s (High &rarr; Low)</option>
<option value="prefill">Prompt / Tokens/s (High &rarr; Low)</option>
<option value="date">Newest Date</option>
</select>
</div>
</div>

<div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-3">
<div class="oflm-pill-tabs" id="cb-view-tabs" role="tablist" aria-label="Leaderboard views">
<button type="button" class="active" data-cb-tab="leaderboard">Runs Table</button>
<button type="button" data-cb-tab="models">Model Comparison</button>
<button type="button" data-cb-tab="hardware">Hardware Comparison</button>
<button type="button" data-cb-tab="trends">Stage Scaling</button>
</div>
<div class="d-flex align-items-center gap-3 small" style="color: var(--oflm-muted);">
<span id="cb-count">Loading runs...</span>
<button type="button" id="cb-reset" class="btn btn-sm btn-link p-0 text-decoration-none">Reset filters</button>
</div>
</div>

<div id="cb-panel-leaderboard">
<div class="oflm-table-wrap">
<table class="oflm-table">
<thead>
<tr>
<th>Model</th>
<th>Capability</th>
<th>Hardware &amp; RAM</th>
<th>Runtime</th>
<th>Quantization</th>
<th>Prompt / Tok/s</th>
<th>Decode / Txt/s</th>
<th>First Token / Batch Time</th>
<th>Source &amp; Date</th>
</tr>
</thead>
<tbody id="cb-rows">
<tr><td colspan="9">Loading benchmark runs...</td></tr>
</tbody>
</table>
</div>
</div>

<div id="cb-panel-models" class="d-none">
<div class="oflm-chart-card">
<div class="oflm-chart-header">
<div>
<h3 class="oflm-chart-title">Headline Speed by Model</h3>
<p class="oflm-chart-subtitle">Includes repository sweeps and averaged community runs (logarithmic scale)</p>
</div>
<div class="oflm-pill-tabs" id="cb-models-metric">
<button type="button" class="active" data-metric="decode">Decode / Texts/s</button>
<button type="button" data-metric="prefill">Prompt / Tokens/s</button>
<button type="button" data-metric="ttft">Latency / First Token (ms)</button>
</div>
</div>
<div id="cb-chart-models" class="oflm-echarts-canvas" style="width: 100%; height: 380px;" aria-label="Model throughput comparison chart"></div>
</div>
</div>

<div id="cb-panel-hardware" class="d-none">
<div class="oflm-chart-card">
<div class="oflm-chart-header">
<div>
<h3 class="oflm-chart-title">Hardware Platform Breakdown</h3>
<p class="oflm-chart-subtitle">Mean and peak throughput by Ryzen AI processor (logarithmic scale)</p>
</div>
</div>
<div id="cb-chart-hardware" class="oflm-echarts-canvas" style="width: 100%; height: 380px;" aria-label="Hardware comparison chart"></div>
</div>
</div>

<div id="cb-panel-trends" class="d-none">
<div class="oflm-chart-card">
<div class="oflm-chart-header">
<div>
<h3 class="oflm-chart-title">Multi-Stage Scaling Curves</h3>
<p class="oflm-chart-subtitle">Context stages (1k-32k) or batch sizes (1-128) on a logarithmic throughput scale</p>
</div>
<div class="oflm-pill-tabs" id="cb-trends-mode">
<button type="button" class="active" data-mode="llm">Chat Context Scaling</button>
<button type="button" data-mode="embed">Embedding Batch Scaling</button>
</div>
</div>
<div id="cb-chart-trends" class="oflm-echarts-canvas" style="width: 100%; height: 380px;" aria-label="Multi-stage scaling trends chart"></div>
</div>
</div>

</div>

## Run & Share Benchmarks from Your PC

Plug into wall power, close heavy background apps, and run `oflm bench` or `oflm bench-embed` with `--submit`:

```bash
# Measure chat prompt and generation speed across context lengths + upload
oflm bench qwen3.5:4b --submit

# Measure batched embedding throughput across batch sizes + upload
oflm bench-embed bge-base:en-v1.5 --max-batch 128 --bench-iterations 3 --submit
```
