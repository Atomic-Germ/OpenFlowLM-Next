---
layout: page
title: "Benchmark Detail"
permalink: /benchmarks/community/detail/
description: "Single community benchmark — system info, per-stage results."
---

<link rel="stylesheet" href="{{ '/assets/community-bench.css' | relative_url }}">
<div data-cb-page="detail" class="cb-wrap">
  <p><a href="{{ '/benchmarks/community/' | relative_url }}">Benchmarks</a> › <span id="cb-crumb-model">…</span> › <span id="cb-crumb-id"></span>
     <span class="cb-validated">Validated</span></p>
  <h1 id="cb-title">Benchmark</h1>
  <div id="cb-detail"><p class="cb-note">Loading… (pass ?id=000142)</p></div>
</div>
<script>document.documentElement.dataset.cbApi = {{ site.cb_api | default: '"/api"' | jsonify }};</script>
<script src="{{ '/assets/community-bench.js' | relative_url }}"></script>
<script>
  // keep breadcrumbs in sync with the router in community-bench.js
  addEventListener("DOMContentLoaded", () => {
    const id = new URLSearchParams(location.search).get("id") || location.hash.slice(1);
    document.getElementById("cb-crumb-id").textContent = id ? "#" + id : "";
    document.getElementById("cb-title").textContent = "Benchmark " + (id || "");
  });
</script>
