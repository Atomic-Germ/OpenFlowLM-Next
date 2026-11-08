// Pure, dependency-free parsing + validation for `oflm bench` CSV output.
// Mirrors src/src/benchmarking.hpp (write_bench_csv) and
// src/src/benchmark_embed.hpp (write_embed_bench_csv). Unit-tested in test/.

export function parseBenchCsv(text) {
  const meta = {};
  const lines = String(text).split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  let headerIdx = -1;
  lines.forEach((l, i) => {
    if (l.startsWith("#")) {
      const m = l.replace(/^#\s*/, "");
      const eq = m.indexOf("=");
      if (eq > 0) meta[m.slice(0, eq).trim()] = m.slice(eq + 1).trim();
    } else if (l.startsWith("context_length_k,") || l.startsWith("batch,")) headerIdx = i;
  });
  if (headerIdx < 0) throw new Error("csv_missing_header");
  const headers = lines[headerIdx].split(",").map((h) => h.trim());
  const rows = [];
  for (let i = headerIdx + 1; i < lines.length; i++) {
    const cells = lines[i].split(",");
    if (cells.length < 2) continue;
    const row = {};
    headers.forEach((h, j) => (row[h] = (cells[j] ?? "").trim()));
    rows.push(row);
  }
  if (!rows.length) throw new Error("csv_no_rows");
  return { meta, headers, rows };
}

function f(row, key) {
  const v = row[key];
  if (v === "" || v == null) return null;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error("csv_bad_number:" + key);
  return n;
}

function checkStat(avg, std, min, max, label) {
  for (const [v, name] of [[avg, "avg"], [std, "std"], [min, "min"], [max, "max"]]) {
    if (v == null) continue;
    if (v < 0) throw new Error("csv_negative:" + label + "_" + name);
  }
  if (avg != null && min != null && avg < min - 1e-9) throw new Error("csv_avg_below_min:" + label);
  if (avg != null && max != null && avg > max + 1e-9) throw new Error("csv_avg_above_max:" + label);
  if (min != null && max != null && min > max) throw new Error("csv_min_above_max:" + label);
}

export function stagesFromCsv(csvText) {
  const { meta, headers, rows } = parseBenchCsv(csvText);
  const isEmbed = headers[0] === "batch";
  const stages = rows.map((r, i) => {
    if (isEmbed) {
      const batch = f(r, "batch");
      if (batch == null || !Number.isInteger(batch) || batch < 1 || (batch & (batch - 1)) !== 0)
        throw new Error("embed_bad_batch");
      return {
        ctx_k: batch, // keep column name; UI labels it "batch" for embeddings
        ttft_avg_s: f(r, "batched_avg_s"),
        ttft_std_s: f(r, "batched_std_s"),
        ttft_min_s: f(r, "batched_min_s"),
        ttft_max_s: f(r, "batched_max_s"),
        prefill_avg: f(r, "tokens_avg_per_s"),
        prefill_std: f(r, "tokens_std_per_s"),
        prefill_min: f(r, "tokens_min_per_s"),
        prefill_max: f(r, "tokens_max_per_s"),
        decode_avg: f(r, "texts_avg_per_s"),
        decode_std: f(r, "texts_std_per_s"),
        decode_min: f(r, "texts_min_per_s"),
        decode_max: f(r, "texts_max_per_s"),
        looped_avg_s: f(r, "looped_avg_s"),
        speedup: f(r, "batch_speedup"),
      };
    }
    const ctx = f(r, "context_length_k");
    if (ctx == null || !Number.isInteger(ctx) || ctx < 1 || (ctx & (ctx - 1)) !== 0)
      throw new Error("ctx_not_power_of_two");
    checkStat(f(r, "ttft_avg_s"), f(r, "ttft_std_s"), f(r, "ttft_min_s"), f(r, "ttft_max_s"), "ttft");
    checkStat(f(r, "prefill_avg_toks_per_s"), f(r, "prefill_std_toks_per_s"),
              f(r, "prefill_min_toks_per_s"), f(r, "prefill_max_toks_per_s"), "prefill");
    checkStat(f(r, "decoding_avg_toks_per_s"), f(r, "decoding_std_toks_per_s"),
              f(r, "decoding_min_toks_per_s"), f(r, "decoding_max_toks_per_s"), "decode");
    return {
      ctx_k: ctx,
      ttft_avg_s: f(r, "ttft_avg_s"), ttft_std_s: f(r, "ttft_std_s"),
      ttft_min_s: f(r, "ttft_min_s"), ttft_max_s: f(r, "ttft_max_s"),
      prefill_avg: f(r, "prefill_avg_toks_per_s"), prefill_std: f(r, "prefill_std_toks_per_s"),
      prefill_min: f(r, "prefill_min_toks_per_s"), prefill_max: f(r, "prefill_max_toks_per_s"),
      decode_avg: f(r, "decoding_avg_toks_per_s"), decode_std: f(r, "decoding_std_toks_per_s"),
      decode_min: f(r, "decoding_min_toks_per_s"), decode_max: f(r, "decoding_max_toks_per_s"),
    };
  });
  for (let i = 1; i < stages.length; i++)
    if (stages[i].ctx_k <= stages[i - 1].ctx_k) throw new Error("csv_not_ascending");
  return { isEmbed, meta, stages };
}

export const TASK_VALUES = ["LLM", "Embedding", "Vision-Language", "Audio", "ImageGen"];
export const UUID_V4_RE = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$/;
const PRINT_RE = /^[\x20-\x7E]{0,200}$/;
export function validateSubmission(body) {
  const errs = [];
  if (!body || typeof body !== "object") return ["body_required"];
  if (!body.model || String(body.model).length > 120) errs.push("model_required_120");
  if (!body.submission_id || String(body.submission_id).length > 80) errs.push("submission_id_required");
  if (!body.csv || String(body.csv).length > 200_000) errs.push("csv_required_200k");
  if (body.task && !TASK_VALUES.includes(body.task)) errs.push("task_invalid");
  if (body.ram_gb != null && !(body.ram_gb >= 0 && body.ram_gb <= 4000)) errs.push("ram_invalid");
  // Trusted-key shape: install ids come from oflm bench --submit (uuid v4) or
  // the web form (uuid). Anything else is not from an oflm install.
  if (body.install_id != null && !UUID_V4_RE.test(String(body.install_id))) errs.push("install_id_invalid");
  if (body.survey_opt_in != null && body.survey_opt_in !== true && body.survey_opt_in !== false)
    errs.push("survey_invalid");
  for (const k of ["cpu", "hardware", "os", "backend", "quantization"]) {
    if (body[k] != null && !PRINT_RE.test(String(body[k]))) errs.push("field_invalid:" + k);
  }
  let parsed = null;
  if (!errs.includes("csv_required_200k")) {
    try { parsed = stagesFromCsv(body.csv); }
    catch (e) { errs.push(e.message); }
  }
  return { errs, parsed };
}
