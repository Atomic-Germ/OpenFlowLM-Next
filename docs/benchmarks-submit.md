---
layout: page
title: "Submit a Benchmark"
permalink: /benchmarks/submit/
description: "Submit an oflm bench result to the community leaderboard."
---

<link rel="stylesheet" href="{{ '/assets/community-bench.css' | relative_url }}">
<div data-cb-page="submit" class="cb-wrap cb-form">
  <h1>Submit a Benchmark</h1>
  <p class="cb-note">1. Run <code>oflm bench &lt;model&gt;</code> (writes <code>bench_&lt;tag&gt;_&lt;date&gt;.csv</code>). 2. Paste or drop that CSV below. 3. Fill in your hardware. The server recomputes medians and marks valid rows <b>Validated</b>.</p>
  <form id="cb-submit">
    <label>Model<input id="s-model" required placeholder="Qwen3.6-35B-A3B"></label>
    <label>Task<select id="s-task"><option>LLM</option><option>Embedding</option><option>Vision-Language</option><option>Audio</option><option>ImageGen</option></select></label>
    <label>Hardware (GPU/NPU)<input id="s-hardware" required placeholder="Radeon 890M (Ryzen AI HX 370)"></label>
    <label>CPU<input id="s-cpu" placeholder="AMD Ryzen AI 9 HX 370 (12 cores)"></label>
    <label>System Memory (GB)<input id="s-ram" type="number" min="1" max="4000" placeholder="96"></label>
    <label>Backend<input id="s-backend" placeholder="NPU (XDNA2) / Vulkan / Hybrid"></label>
    <label>Quantization<input id="s-quant" placeholder="Q4_K_M"></label>
    <label>OFLM Version<input id="s-version" placeholder="0.2.0"></label>
    <label>OS<input id="s-os" placeholder="Windows 11 24H2"></label>
    <label>Runs (iterations)<input id="s-runs" type="number" min="0" placeholder="5"></label>
    <label>bench CSV<textarea id="s-csv" rows="8" required placeholder="context_length_k,ttft_avg_s,…"></textarea></label>
    <button>Submit</button> <span id="cb-msg" class="cb-note"></span>
  </form>
  <p class="cb-note">Trusted-key only: the form mints a 10-minute HMAC token from the Worker (<code>GET /api/token</code>) and the server validates schema + ranges. Raw CSV is stored for audit. Duplicate <code>submission_id</code> is rejected.</p>
</div>
<script>document.documentElement.dataset.cbApi = {{ site.cb_api | default: '"/api"' | jsonify }};</script>
<script src="{{ '/assets/community-bench.js' | relative_url }}"></script>
