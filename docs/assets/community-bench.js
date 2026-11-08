// Community benchmarks frontend — leaderboard / detail / submit.
// API base comes from <meta name="cb-api" content="https://oflm-bench.…"> in
// _config.yml `cb_api` or window.CB_API. Falls back to /api (local Pages).
(async () => {
  const API = document.documentElement.dataset.cbApi || "/api";
  const pageEl = document.querySelector("[data-cb-page]");
  if (!pageEl) return;
  const kind = pageEl.dataset.cbPage;
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---------- leaderboard ----------
  async function loadLeaderboard() {
    const q = new URLSearchParams({
      model: $("#f-model")?.value || "", task: $("#f-task")?.value || "",
      hardware: $("#f-hardware")?.value || "", backend: $("#f-backend")?.value || "",
      quantization: $("#f-quant")?.value || "", oflm_version: $("#f-version")?.value || "",
      sort: $("#f-sort")?.value || "decode", page: "1", size: "20", status: "validated",
    });
    for (const [k, v] of [...q]) if (!v) q.delete(k);
    const res = await fetch(`${API}/benchmarks?${q}`, { mode: "cors" }).catch(() => null);
    const body = res?.ok ? await res.json() : { results: [], total: 0, error: "api_unreachable" };
    const rows = body.results || [];
    const table = $("#cb-rows");
    if (table) table.innerHTML = rows.map((r) => `
      <tr onclick="location.href='/benchmarks/community/detail/?id=${esc(r.id)}'">
        <td><strong>${esc(r.model)}</strong></td><td><span class="cb-badge ${esc(r.task)}">${esc(r.task)}</span></td>
        <td>${esc(r.hardware)}</td><td>${esc(r.ram_gb ?? "")}${r.ram_gb ? " GB" : ""}</td>
        <td>${esc(r.backend)}</td><td>${esc(r.quantization)}</td>
        <td>${esc(r.headline.prefill ?? "—")}</td><td><strong>${esc(r.headline.decode ?? "—")}</strong></td>
        <td>${esc(r.headline.ttft_ms ?? "—")}</td><td>${esc(r.oflm_version)}</td><td>${esc(r.date)}</td><td>›</td></tr>`).join("")
      || `<tr><td colspan="12">${esc(body.error || "No results")} — is the API deployed? See docs/DEPLOY-CLOUDFLARE.md.</td></tr>`;
    const count = $("#cb-count"); if (count) count.textContent = `Showing 1–${rows.length} of ${body.total ?? 0} results`;
    const cards = $("#cb-cards");
    if (cards) cards.innerHTML = rows.map((r) => `
      <div class="cb-card"><strong>${esc(r.model)}</strong> <span class="cb-badge">${esc(r.task)}</span>
      <div>${esc(r.hardware)} · ${esc(r.backend)}</div><div>Decode <b>${esc(r.headline.decode ?? "—")} tok/s</b></div>
      <div class="cb-note">${esc(r.date)} · OFLM ${esc(r.oflm_version)}</div></div>`).join("");
    ["f-model", "f-task", "f-hardware", "f-backend", "f-quant", "f-version", "f-sort"].forEach((id) =>
      document.getElementById(id)?.addEventListener("change", loadLeaderboard));
    $("#cb-mode-table")?.addEventListener("click", () => { $("#cb-tablewrap").style.display = ""; $("#cb-cards").style.display = "none"; });
    $("#cb-mode-cards")?.addEventListener("click", () => { $("#cb-tablewrap").style.display = "none"; $("#cb-cards").style.display = ""; });
  }

  // ---------- detail ----------
  async function loadDetail(id) {
    const res = await fetch(`${API}/benchmarks/${encodeURIComponent(id)}`, { mode: "cors" }).catch(() => null);
    if (!res?.ok) { $("#cb-detail").innerHTML = `<p class="cb-note">Benchmark #${esc(id)} not found (API: ${esc(API)}).</p>`; return; }
    const d = await res.json(), h = d.headline || {};
    const isEmbed = d.task === "Embedding";
    const stages = d.stages || [];
    // Embedding numbers are batch-sweep numbers, not prefill/decode: batch is
    // the axis, TTFT column is batched latency, "prefill" is tokens/s and
    // "decode" is texts/s; looped + speedup show what batching buys.
    const metricHtml = isEmbed
      ? `<div class="cb-metric">Texts Throughput<b>${esc(h.decode ?? "—")}</b><span class="cb-note">texts/sec (headline)</span></div>
         <div class="cb-metric">Tokens Throughput<b>${esc(h.prefill ?? "—")}</b><span class="cb-note">tokens/sec (headline)</span></div>
         <div class="cb-metric">Batched Latency<b>${esc(h.ttft_ms ?? "—")}</b><span class="cb-note">ms (headline batch)</span></div>
         <div class="cb-metric">Batch Stage<b>${esc(h.ctx_k ?? "—")}</b><span class="cb-note">headline batch size</span></div>`
      : `<div class="cb-metric">Decode Throughput<b>${esc(h.decode ?? "—")}</b><span class="cb-note">tokens/sec</span></div>
         <div class="cb-metric">Prefill Throughput<b>${esc(h.prefill ?? "—")}</b><span class="cb-note">tokens/sec</span></div>
         <div class="cb-metric">Time to First Token<b>${esc(h.ttft_ms ?? "—")}</b><span class="cb-note">ms</span></div>
         <div class="cb-metric">Context Stage<b>${esc(h.ctx_k ?? "—")}k</b><span class="cb-note">headline stage</span></div>`;
    const rowHtml = stages.map((s) => isEmbed
      ? `<tr><td>batch ${esc(s.ctx_k)}</td><td>${esc(s.prefill_avg ?? "—")}</td><td>${esc(s.decode_avg ?? "—")}</td><td>${esc(s.ttft_avg_s ?? "—")}</td><td>${esc(s.looped_avg_s ?? "—")}</td><td>${esc(s.speedup ?? "—")}</td></tr>`
      : `<tr><td>${esc(s.ctx_k)}k</td><td>${esc(s.prefill_avg ?? "—")}</td><td>${esc(s.decode_avg ?? "—")}</td><td>${esc(s.ttft_avg_s ?? "—")}</td></tr>`).join("");
    const headHtml = isEmbed
      ? `<tr><th>Batch</th><th>Tokens (tok/s)</th><th>Texts (texts/s)</th><th>Batched (s)</th><th>Looped (s)</th><th>Speedup</th></tr>`
      : `<tr><th>Context</th><th>Prefill (tok/s)</th><th>Decode (tok/s)</th><th>TTFT (s)</th></tr>`;
    const surveyNote = d.survey ? `<tr><td>Hardware survey</td><td>included (anonymous — counts as one machine in the survey)</td></tr>` : "";
    $("#cb-detail").innerHTML = `
      <div class="cb-metrics">${metricHtml}</div>
      <h3>System Information</h3><table class="cb-table">
        <tr><td>CPU</td><td>${esc(d.cpu || d.survey?.cpu || "")}</td></tr><tr><td>Hardware (GPU/NPU)</td><td>${esc(d.hardware)}</td></tr>
        <tr><td>System Memory</td><td>${esc(d.ram_gb ?? d.survey?.ram_gb ?? "")} GB</td></tr><tr><td>Backend</td><td>${esc(d.backend)}</td></tr>
        <tr><td>OS</td><td>${esc(d.os || d.survey?.os || "")}</td></tr><tr><td>OpenFlowLM</td><td>v${esc(d.oflm_version)}</td></tr>
        ${surveyNote}</table>
      <h3>Run Results (per ${isEmbed ? "batch stage" : "context stage"})</h3><table class="cb-table">
        ${headHtml}${rowHtml}
        <tr><td><strong>Median (headline)</strong></td><td><strong>${esc(h.prefill ?? "—")}</strong></td><td><strong>${esc(h.decode ?? "—")}</strong></td><td><strong>${esc(h.ttft_ms ?? "—")} ms</strong></td></tr></table>`;
  }

  // ---------- submit ----------
  async function submitForm() {
    const form = $("#cb-submit"); if (!form) return;
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const msg = $("#cb-msg"); msg.textContent = "Submitting…";
      const token = await fetch(`${API}/token`).then((r) => r.json()).then((j) => j.token).catch(() => "");
      if (!token) { msg.textContent = "Could not mint a token (API deployed?)"; return; }
      const body = {
        submission_id: crypto.randomUUID(), model: $("#s-model").value, task: $("#s-task").value,
        hardware: $("#s-hardware").value, cpu: $("#s-cpu").value, ram_gb: Number($("#s-ram").value) || null,
        backend: $("#s-backend").value, quantization: $("#s-quant").value,
        oflm_version: $("#s-version").value, os: $("#s-os").value,
        runs: Number($("#s-runs").value) || 0, csv: $("#s-csv").value,
      };
      const res = await fetch(`${API}/benchmarks`, {
        method: "POST", headers: { "Content-Type": "application/json", "X-OFLM-Token": token },
        body: JSON.stringify(body),
      });
      const out = await res.json().catch(() => ({}));
      msg.textContent = res.ok ? `Submitted as #${out.id} (${out.status}) — view it at /benchmarks/community/detail/?id=${out.id}`
                               : `Failed (${res.status}): ${out.error || ""} ${out.details?.join(", ") || ""}`;
    });
  }

  if (kind === "leaderboard") await loadLeaderboard();
  if (kind === "detail") await loadDetail(new URLSearchParams(location.search).get("id") || location.hash.slice(1));
  if (kind === "submit") await submitForm();
})();
