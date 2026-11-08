---
layout: page
title: "Community Benchmarks"
permalink: /benchmarks/community/
description: "Real-world OpenFlowLM performance results submitted by users."
---

<link rel="stylesheet" href="{{ '/assets/community-bench.css' | relative_url }}">
<div data-cb-page="leaderboard" class="cb-wrap">
  <h1>Community Benchmarks</h1>
  <p>Real-world performance results from OpenFlowLM-Next users. Compare models, hardware, and configurations. Results are community-submitted and may vary based on system configuration, workloads, and settings.</p>
  <p><a href="{{ '/benchmarks/submit/' | relative_url }}"><button>Submit a Benchmark</button></a>
     <a href="{{ '/benchmarks/' | relative_url }}">How it works</a></p>

  <div class="cb-tabs"><button class="cb-tab active">Leaderboard</button><button class="cb-tab">Model Comparison</button><button class="cb-tab">Hardware Comparison</button><button class="cb-tab">Trends</button></div>

  <div class="cb-filters">
    <div><label for="f-task">Task</label><select id="f-task"><option value="">All Tasks</option><option>LLM</option><option>Embedding</option><option>Vision-Language</option><option>Audio</option></select></div>
    <div><label for="f-model">Model</label><input id="f-model" placeholder="All Models"></div>
    <div><label for="f-hardware">Hardware (GPU/NPU)</label><input id="f-hardware" placeholder="All Hardware"></div>
    <div><label for="f-backend">Backend</label><input id="f-backend" placeholder="All Backends"></div>
    <div><label for="f-quant">Quantization</label><input id="f-quant" placeholder="All Quantizations"></div>
    <div><label for="f-version">OFLM Version</label><input id="f-version" placeholder="All Versions"></div>
    <div><label for="f-sort">Sort by</label><select id="f-sort"><option value="decode">Decode (tok/s)</option><option value="prefill">Prefill (tok/s)</option><option value="date">Date</option></select></div>
    <div><span id="cb-count" class="cb-note"></span>
      <button id="cb-mode-table">Table</button> <button id="cb-mode-cards">Cards</button></div>
  </div>

  <div id="cb-tablewrap"><table class="cb-table"><thead><tr><th>Model</th><th>Task</th><th>Hardware</th><th>RAM</th><th>Backend</th><th>Quantization</th><th>Prefill (tok/s)</th><th>Decode (tok/s)</th><th>TTFT (ms)</th><th>OFLM</th><th>Date</th><th></th></tr></thead><tbody id="cb-rows"></tbody></table></div>
  <div id="cb-cards" class="cb-cards" style="display:none"></div>
</div>
<script>document.documentElement.dataset.cbApi = {{ site.cb_api | default: '"/api"' | jsonify }};</script>
<script src="{{ '/assets/community-bench.js' | relative_url }}"></script>
