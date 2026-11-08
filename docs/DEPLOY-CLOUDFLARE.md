# Deploying the docs site + community benchmarks to Cloudflare (free tier)

Architecture (matches the answers you gave):

| Piece | Tech | Free-tier budget |
|---|---|---|
| Docs site (`docs/`, Jekyll) | Cloudflare **Pages** (static `_site`) | Unlimited bandwidth, 20k builds/mo |
| Benchmark API | Cloudflare **Worker** (`benchmarks-worker/`) | 100k req/day |
| Benchmark DB | Cloudflare **D1** (`oflm-bench`) | 5 GB, 100k writes/day (SQL filters server-side, no KV race) |

Everything else (forms, charts) is client-side JS (`docs/assets/community-bench.js`),
so new posts don't need redeploys. If OpenFlowLM ever gets popular enough that
100k req/day hurts, put a Cache rule on `GET /api/benchmarks*` first — that alone
absorbs most leaderboard reads.

## 0. Prerequisites

```bash
npx --yes wrangler@latest login   # opens browser, one-time
# Pages build needs Ruby on the build host; GitHub Actions or Cloudflare Pages both work.
```

## 1. Provision D1 (one-time)

```bash
cd benchmarks-worker
cp wrangler.toml wrangler.toml.bak
npx wrangler d1 create oflm-bench          # copy the database_id into wrangler.toml
npx wrangler d1 execute oflm-bench --local --file schema.sql   # dev
npx wrangler d1 execute oflm-bench --remote --file schema.sql  # prod
# (`d1 execute --file`, not `d1 migrations apply --file`: the latter takes a
# migrations/ directory of numbered files and rejects a --file argument.)
npx wrangler secret put TOKEN_SECRET       # 32+ random bytes — mints form tokens
# optional static Bearer for future `oflm bench --submit` CLI work:
# npx wrangler secret put SUBMIT_BEARER
```

## 2. Worker API

```bash
cd benchmarks-worker
npm test            # node --test (no deps)
npx wrangler dev --local   # →  http://127.0.0.1:8787
npx wrangler deploy        # prod → https://oflm-bench.<account>.workers.dev
curl "$API/api/benchmarks?status=validated&sort=decode"
curl "$API/api/benchmarks/000142"
```

Endpoints (`benchmarks-worker/src/index.js`): `GET /api/benchmarks` (filters:
model, task, hardware, backend, quantization, oflm_version, status, sort, page, size),
`GET /api/benchmarks/:id`, `GET /api/stats`, `GET /api/token` (10-min HMAC),
`POST /api/benchmarks` (validated against the real `oflm bench` /
`oflm bench-embed` CSV shape — see `src/benchmarking.hpp` and
`src/benchmark_embed.hpp`), `OPTIONS` (CORS).

`oflm bench --submit` / `oflm bench-embed --submit` POST from the CLI
(libcurl → token → POST). It attaches an anonymous per-install UUID
(`bench_consent.json`, `~/.config/oflm` on Linux, `~/.oflm` on Windows) whose
SHA-256 the server stores as `install_hash` to count distinct machines for the
hardware survey. Survey fields (CPU, RAM, OS, NPU) are sent only when the user
opted in on the first run; the embedding shape keeps batch/texts/tokens/speedup
in `bench_stages` (`ctx_k` = batch size for embeddings) and the detail page
renders the embedding labels instead of prefill/decode.

## 3. Jekyll site → Pages

```bash
cd docs
gem install bundler && bundle install
bundle exec jekyll build          # → docs/_site
npx wrangler pages deploy docs/_site --project-name openflowlm
```

Then point Pages at the zone (openflowlm.org is currently GitHub Pages via
`docs/CNAME`) — set `cb_api` in `docs/_config.yml`:

```yaml
cb_api: "https://oflm-bench.<account>.workers.dev"
```

(or leave it as `/api` and use a Pages Function proxy instead of CORS.)

CI option: GitHub Actions → `bundle exec jekyll build` + `wrangler pages deploy`.
Only run that when you move off GitHub Pages; until then GitHub Pages keeps working
and the Worker is independently live.

## 4. Seeds / daily ops

`benchmarks-worker/schema.sql` seeds three rows mirroring the mockup (#000142…)
so the UI is testable without hardware. To backfill real results, run
`utilities/bench-submit/submit.py` or the `/benchmarks/submit/` form.
