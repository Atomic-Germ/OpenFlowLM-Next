// OpenFlowLM community benchmarks — Worker API (D1, free tier).
// Routes: GET /api/benchmarks, GET /api/benchmarks/:id, GET /api/stats,
//         GET /api/token, POST /api/benchmarks, OPTIONS *.
// Docs: benchmarks-worker/README.md
import { stagesFromCsv, validateSubmission } from "./parse.js";
import { mintToken, isAuthorized } from "./auth.js";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type,X-OFLM-Token,Authorization",
};
function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status, headers: { "Content-Type": "application/json", ...CORS },
  });
}
const fail = (code, status = 400) => json({ error: code }, status);

function publicRow(r) {
  return {
    id: r.public_id, model: r.model, task: r.task, hardware: r.hardware,
    cpu: r.cpu, ram_gb: r.ram_gb, backend: r.backend, quantization: r.quantization,
    oflm_version: r.oflm_version, os: r.os, config: safeJson(r.config),
    runs: r.runs, status: r.status, source: r.source,
    survey: r.survey ? safeJson(r.survey) : null,
    headline: { ctx_k: r.headline_ctx_k, prefill: r.prefill_tok_s, decode: r.decode_tok_s, ttft_ms: r.ttft_ms },
    date: r.report_date,
  };
}
const safeJson = (s) => { try { return JSON.parse(s); } catch { return {}; } };

export async function handleGetBenchmarks(url, env) {
  const q = url.searchParams;
  const page = Math.max(1, Number(q.get("page")) || 1);
  const size = Math.min(100, Math.max(1, Number(q.get("size")) || 20));
  const sort = q.get("sort") || "date";
  const WHERE = [], BIND = [];
  const add = (col, key) => { if (q.get(key)) { WHERE.push(`${col} = ?`); BIND.push(q.get(key)); } };
  add("model", "model"); add("task", "task"); add("hardware", "hardware");
  add("backend", "backend"); add("quantization", "quantization"); add("oflm_version", "oflm_version");
  if (q.get("status")) { WHERE.push("status = ?"); BIND.push(q.get("status")); }
  const order = { decode: "decode_tok_s DESC", prefill: "prefill_tok_s DESC", date: "report_date DESC, id DESC" }[sort] || "report_date DESC, id DESC";
  const total = await env.DB.prepare(
    `SELECT COUNT(*) AS n FROM benchmarks ${WHERE.length ? "WHERE " + WHERE.join(" AND ") : ""}`).bind(...BIND).first();
  const { results } = await env.DB.prepare(
    `SELECT * FROM benchmarks ${WHERE.length ? "WHERE " + WHERE.join(" AND ") : ""} ORDER BY ${order} LIMIT ? OFFSET ?`)
    .bind(...BIND, size, (page - 1) * size).all();
  return json({ page, size, total: total?.n ?? 0, results: results.map(publicRow) });
}

export async function handleGetOne(id, env) {
  const b = await env.DB.prepare("SELECT * FROM benchmarks WHERE public_id = ?").bind(id).first();
  if (!b) return fail("not_found", 404);
  const { results: stages } = await env.DB.prepare(
    "SELECT * FROM bench_stages WHERE benchmark_id = ? ORDER BY ctx_k").bind(b.id).all();
  return json({ ...publicRow(b), stages });
}

export async function handlePost(req, env) {
  if (!await isAuthorized(req, env)) return fail("unauthorized", 401);
  let body;
  try { body = await req.json(); } catch { return fail("invalid_json"); }
  const { errs, parsed } = validateSubmission(body);
  if (errs.length) return json({ error: "validation_failed", details: errs }, 422);

  const dup = await env.DB.prepare("SELECT 1 FROM benchmarks WHERE submission_id = ? LIMIT 1")
    .bind(String(body.submission_id)).first();
  if (dup) return fail("duplicate", 409);

  const stage1 = parsed.stages.find((s) => s.ctx_k === 1) || parsed.stages[0];
  const status = errs.length ? "pending" : "validated";
  const now = new Date().toISOString();
  // Survey privacy: server keeps only sha256(install_id) for distinct-machine
  // survey counts; the raw id is never stored.
  let install_hash = null;
  if (body.install_id) {
    const h = await crypto.subtle.digest("SHA-256",
      new TextEncoder().encode(String(body.install_id)));
    install_hash = [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 32);
  }
  const survey = body.survey_opt_in === true
    ? JSON.stringify({ cpu: String(body.cpu || ""), ram_gb: body.ram_gb ?? null,
        os: String(body.os || ""), hardware: String(body.hardware || "") })
    : null;
  const info = await env.DB.prepare(
    `INSERT INTO benchmarks (submission_id, model, task, hardware, cpu, ram_gb, backend, quantization,
      oflm_version, os, config, runs, status, source, headline_ctx_k, prefill_tok_s, decode_tok_s, ttft_ms,
      report_date, created_at, raw_csv, install_hash, survey)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
    .bind(String(body.submission_id), String(body.model), body.task || "LLM",
      String(body.hardware || ""), String(body.cpu || ""),
      body.ram_gb == null ? null : Number(body.ram_gb),
      String(body.backend || ""), String(body.quantization || ""),
      String(body.oflm_version || "0.2.0"), String(body.os || ""),
      JSON.stringify(body.config || {}), Number(body.runs || 0), status,
      String(body.source || "web").slice(0, 16),
      stage1.ctx_k, stage1.prefill_avg ?? null, stage1.decode_avg ?? null,
      stage1.ttft_avg_s == null ? null : Math.round(stage1.ttft_avg_s * 1000),
      String(body.report_date || now.slice(0, 10)), now, String(body.csv).slice(0, 200_000),
      install_hash, survey)
    .run();

  const stmt = env.DB.prepare(
    `INSERT INTO bench_stages (benchmark_id, ctx_k, ttft_avg_s, ttft_std_s, ttft_min_s, ttft_max_s,
      prefill_avg, prefill_std, prefill_min, prefill_max, decode_avg, decode_std, decode_min, decode_max,
      looped_avg_s, speedup)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
  await env.DB.batch(parsed.stages.map((s) => stmt.bind(
    info.meta.last_row_id, s.ctx_k, s.ttft_avg_s ?? null, s.ttft_std_s ?? null, s.ttft_min_s ?? null, s.ttft_max_s ?? null,
    s.prefill_avg ?? null, s.prefill_std ?? null, s.prefill_min ?? null, s.prefill_max ?? null,
    s.decode_avg ?? null, s.decode_std ?? null, s.decode_min ?? null, s.decode_max ?? null,
    s.looped_avg_s ?? null, s.speedup ?? null)));

  const row = await env.DB.prepare("SELECT public_id FROM benchmarks WHERE id = ?").bind(info.meta.last_row_id).first();
  return json({ id: row?.public_id, status }, 201);
}

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    if (req.method === "OPTIONS") return new Response(null, { headers: CORS });
    if (!env.DB) return fail("db_unbound", 500);
    if (req.method === "GET" && url.pathname === "/api/benchmarks") return handleGetBenchmarks(url, env);
    const one = url.pathname.match(/^\/api\/benchmarks\/([A-Za-z0-9-]+)$/);
    if (req.method === "GET" && one) return handleGetOne(one[1], env);
    if (req.method === "GET" && url.pathname === "/api/stats") {
      const s = await env.DB.prepare(
        `SELECT COUNT(*) n, COUNT(DISTINCT model) m, COUNT(DISTINCT hardware) h
         FROM benchmarks WHERE status='validated'`).first();
      return json(s);
    }
    if (req.method === "GET" && url.pathname === "/api/token") {
      if (!env.TOKEN_SECRET) return fail("auth_not_configured", 500);
      return json({ token: await mintToken(env.TOKEN_SECRET, Number(env.TOKEN_TTL_S || 600)) });
    }
    if (req.method === "POST" && url.pathname === "/api/benchmarks") return handlePost(req, env);
    return fail("not_found", 404);
  },
};
