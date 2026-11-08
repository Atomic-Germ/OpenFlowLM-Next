-- OpenFlowLM community benchmarks — D1 schema (free tier: 5GB, 100k rows? unlimited rows, 100k writes/day)
-- One row per `oflm bench` submission; per-context stages in a child table so the
-- leaderboard can sort on the 1k stage while detail pages show the full sweep.

CREATE TABLE IF NOT EXISTS benchmarks (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  public_id             TEXT NOT NULL UNIQUE,          -- e.g. '000142' (zero-padded id)
  submission_id         TEXT NOT NULL UNIQUE,          -- client-generated uuid, idempotency key
  model                 TEXT NOT NULL,                 -- e.g. 'Qwen3.6-35B-A3B'
  task                  TEXT NOT NULL DEFAULT 'LLM',   -- LLM | Embedding | Vision-Language | Audio | ImageGen
  hardware              TEXT NOT NULL DEFAULT '',      -- free text, e.g. 'Radeon 890M (Ryzen AI HX 370)'
  cpu                   TEXT NOT NULL DEFAULT '',
  ram_gb                REAL,
  backend               TEXT NOT NULL DEFAULT '',      -- NPU (XDNA2) | Vulkan | Hybrid | NPU ...
  quantization          TEXT NOT NULL DEFAULT '',      -- Q4_K_M, Q8, FP16 ...
  oflm_version          TEXT NOT NULL DEFAULT '0.2.0',
  os                    TEXT NOT NULL DEFAULT '',
  config                TEXT NOT NULL DEFAULT '{}',    -- JSON: prompt len, gen len, ctx, sampling, iterations
  runs                  INTEGER NOT NULL DEFAULT 0,    -- measured iterations
  status                TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','validated','rejected')),
  source                TEXT NOT NULL DEFAULT 'web',   -- web | cli | seed
  headline_ctx_k        INTEGER NOT NULL DEFAULT 1,    -- stage used for leaderboard sort (1k)
  prefill_tok_s         REAL,                          -- headline stage aggregates
  decode_tok_s          REAL,
  ttft_ms               REAL,
  report_date           TEXT NOT NULL DEFAULT (date('now')),
  created_at            TEXT NOT NULL DEFAULT (datetime('now')),
  raw_csv               TEXT,                           -- original CSV text, for audit
  install_hash          TEXT,                           -- sha256(install_id), survey de-dup only
  survey                TEXT                            -- JSON hardware survey, only when opted in
);
CREATE INDEX IF NOT EXISTS idx_bench_model   ON benchmarks(model);
CREATE INDEX IF NOT EXISTS idx_bench_task    ON benchmarks(task);
CREATE INDEX IF NOT EXISTS idx_bench_hw      ON benchmarks(hardware);
CREATE INDEX IF NOT EXISTS idx_bench_date    ON benchmarks(report_date DESC);
CREATE INDEX IF NOT EXISTS idx_bench_decode  ON benchmarks(decode_tok_s DESC);
CREATE INDEX IF NOT EXISTS idx_bench_status  ON benchmarks(status);

CREATE TABLE IF NOT EXISTS bench_stages (
  benchmark_id          INTEGER NOT NULL REFERENCES benchmarks(id) ON DELETE CASCADE,
  ctx_k                 INTEGER NOT NULL,              -- 1,2,4,8,16,32
  ttft_avg_s            REAL, ttft_std_s REAL, ttft_min_s REAL, ttft_max_s REAL,
  prefill_avg           REAL, prefill_std REAL, prefill_min REAL, prefill_max REAL,
  decode_avg            REAL, decode_std  REAL, decode_min  REAL, decode_max  REAL,
  looped_avg_s          REAL,                           -- embed only: looped-path cost
  speedup               REAL,                           -- embed only: looped/batched ratio
  PRIMARY KEY (benchmark_id, ctx_k)
);

-- Existing DBs created before this column set: run once (D1 ignores dup ALTERs
-- only via error, so gate on profile in a scratch run, or drop+re-migrate for dev):
--   ALTER TABLE benchmarks ADD COLUMN install_hash TEXT;
--   ALTER TABLE benchmarks ADD COLUMN survey TEXT;
--   ALTER TABLE bench_stages ADD COLUMN looped_avg_s REAL;
--   ALTER TABLE bench_stages ADD COLUMN speedup REAL;

-- Local dev seed (matches mockup #000142 shape so UI copy is verifiable without hardware).
INSERT OR IGNORE INTO benchmarks
  (id, public_id, submission_id, model, task, hardware, cpu, ram_gb, backend, quantization,
   oflm_version, os, runs, status, source, headline_ctx_k, prefill_tok_s, decode_tok_s, ttft_ms, report_date)
VALUES
  (1,'000142','seed-000142','Qwen3.6-35B-A3B','LLM','Radeon 890M (Ryzen AI HX 370)','AMD Ryzen AI 9 HX 370 (12 cores)',96,'Vulkan','Q4_K_M','0.2.0','Windows 11 24H2',5,'validated','seed',1,820,34.7,210,'2026-10-14'),
  (2,'000141','seed-000141','Qwen3-8B','LLM','Radeon 780M (Ryzen 7 7840U)','AMD Ryzen 7 7840U',32,'Vulkan','Q4_K_M','0.2.0','Windows 11',5,'validated','seed',1,1240,62.1,120,'2026-10-12'),
  (3,'000140','seed-000140','Gemma3-4B','LLM','Radeon 890M','AMD Ryzen AI 9 HX 370',64,'Hybrid','Q4_K_M','0.2.0','Windows 11',5,'validated','seed',1,1580,51.3,145,'2026-10-11');

INSERT OR IGNORE INTO bench_stages (benchmark_id, ctx_k, ttft_avg_s, prefill_avg, decode_avg, decode_std)
VALUES (1,1,0.210,820,34.7,4.1), (1,2,0.410,790,33.1,3.8), (1,4,0.820,760,31.9,4.3),
       (2,1,0.120,1240,62.1,5.2), (3,1,0.145,1580,51.3,3.9);
