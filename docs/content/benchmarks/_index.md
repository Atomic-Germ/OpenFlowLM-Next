---
title: "Benchmarks & Hardware Leaderboard"
description: "AMD Ryzen AI NPU throughput across models, quantizations, and memory configurations, plus community hardware runs."
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

<div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-2">
<div class="d-flex flex-wrap align-items-center gap-2">
<h2 class="h4 fw-bold mb-0">Throughput by Model, Quantization &amp; Memory</h2>
{{< b k="wip" issue="240" t="Expanded Family Sweeps #240" >}}
</div>
</div>
<p class="small mb-3" style="color: var(--oflm-muted);">Pick a model and quantization below. Each NPU row plots the median dot and min-max range across RAM speeds and power modes. Hover a row for min and max configs, or click the matrix to highlight a memory tier.</p>

{{< all-runs-chart view="full" >}}

<div class="mt-4 pt-2" data-cb-page="leaderboard">

<div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-3">
<div>
<h2 class="h4 fw-bold mb-1">Runs &amp; Community Leaderboard</h2>
<p class="small mb-0" style="color: var(--oflm-muted);">Click any row for per-stage context or batch curves and system specs.</p>
</div>
<a href="submit/" class="btn btn-sm btn-primary"><span class="material-icons align-middle size-18 me-1">upload_file</span>Submit Run</a>
</div>

<div id="cb-seed-banner" class="alert alert-info d-none mb-3" role="status">
<div class="small mb-0"><strong>Verified Runs &amp; Community Submissions:</strong> This static preview shows repository sweeps and averaged community runs. Live submissions load on <code>openflowlm.org</code>. Upload your speeds with <code>oflm bench &lt;tag&gt; --submit</code> or <a href="submit/">Submit Run</a>.</div>
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

<div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-2">
<div class="small" style="color: var(--oflm-muted);" id="cb-count">Loading runs...</div>
<button type="button" id="cb-reset" class="btn btn-sm btn-link p-0 text-decoration-none">Reset filters</button>
</div>

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

## Run & Share Benchmarks from Your PC

Plug into wall power, close heavy background apps, and run `oflm bench` or `oflm bench-embed` with `--submit`:

```bash
# Measure chat prompt and generation speed across context lengths + upload
oflm bench qwen3.5:4b --submit

# Measure batched embedding throughput across batch sizes + upload
oflm bench-embed bge-base:en-v1.5 --max-batch 128 --bench-iterations 3 --submit
```
