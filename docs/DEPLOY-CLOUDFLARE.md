# Deploying OpenFlowLM Docs and Community Benchmarks

OpenFlowLM serves static documentation built with Hugo Extended (`v0.147.9`, Lotus Docs + Apache ECharts) alongside a serverless leaderboard API backed by Cloudflare Workers and D1.

## Architecture Overview

| Component | Technology | Source Directory | Output / Binding |
|---|---|---|---|
| Website and docs | Hugo Extended `0.147.9` on Cloudflare Pages (or GitHub Pages) | `docs/` | `docs/public/` |
| Community Benchmarks API | Cloudflare Worker (`/api/*`) | `benchmarks-worker/` | `oflm-bench` Worker |
| Benchmark database | Cloudflare D1 (SQLite) | `benchmarks-worker/schema.sql` | `DB` (`oflm-bench`) |

Interactive benchmark charts (`static/js/echarts.min.js` and `static/js/oflm-charts.js`) and leaderboard views (`static/js/community-bench.js`) run in the browser, so new benchmark submissions appear without rebuilding the static site.

## 1. Local Build Verification

Run the documentation checks and build `docs/public/` before deploying:

```bash
python3 utilities/check_embed_bench_docs.py
python3 utilities/check_docs_quality.py
./docs/serve-docs.sh --build
```

## 2. Cloudflare Pages Configuration (`openflowlm.org`)

Connect the GitHub repository to a Cloudflare Pages project with these build settings:

| Setting | Value |
|---|---|
| Framework preset | Hugo |
| Root directory | `docs` |
| Build command | `hugo --minify` |
| Build output directory | `public` |
| Environment variable `HUGO_VERSION` | `0.147.9` |

To deploy directly from a workstation or CI job with Wrangler:

```bash
./docs/serve-docs.sh --build
npx --yes wrangler@latest pages deploy docs/public --project-name openflowlm
```

## 3. Provisioning Cloudflare D1 and `benchmarks-worker/`

Authenticate Wrangler once, create the D1 database, apply `schema.sql`, and store the HMAC token secret:

```bash
npx --yes wrangler@latest login
npx --yes wrangler@latest d1 create oflm-bench
```

Copy the returned `database_id` into `benchmarks-worker/wrangler.toml`, then initialize local and remote tables and deploy the Worker:

```bash
npx --yes wrangler@latest --cwd benchmarks-worker d1 execute oflm-bench --local --file schema.sql
npx --yes wrangler@latest --cwd benchmarks-worker d1 execute oflm-bench --remote --file schema.sql
npx --yes wrangler@latest --cwd benchmarks-worker secret put TOKEN_SECRET
npm --prefix benchmarks-worker test
npx --yes wrangler@latest --cwd benchmarks-worker deploy
```

> [!NOTE]
> Use `d1 execute --file schema.sql` rather than `d1 migrations apply`; the migrations subcommand expects a numbered `migrations/` directory. Optionally set `SUBMIT_BEARER` via `wrangler secret put SUBMIT_BEARER` if static bearer authentication is needed.

### Routing `/api/*` and CLI Traffic

`docs/hugo.toml` sets `params.cb_api = "/api"` by default for same-origin browser requests on `openflowlm.org`, while `src/src/bench_submit.hpp` defaults `oflm bench --submit` to `https://oflm-bench.openflowlm.org` (overridable via `--submit-url` or `OFLM_BENCH_SUBMIT_URL`):

- **Custom domain & routes (recommended):** In Cloudflare Dashboard under **Workers & Pages -> oflm-bench -> Settings -> Domains & Routes**, attach custom domain `oflm-bench.openflowlm.org` and bind the route `openflowlm.org/api/*` to `oflm-bench`. Browser requests to `/api/benchmarks`, `/api/benchmarks/:id`, `/api/stats`, and `/api/token` stay same-origin while CLI `--submit` uploads work out of the box.
- **Separate `workers.dev` origin:** If hosting the static site on another domain while keeping the Worker on `workers.dev`, override `params.cb_api` in `docs/hugo.toml` to `"https://oflm-bench.<account>.workers.dev"`.

Verify the live Worker endpoints after deployment:

```bash
curl --fail "https://openflowlm.org/api/benchmarks?status=validated&sort=decode"
curl --fail "https://openflowlm.org/api/stats"
```

### CLI Submissions (`oflm bench --submit`)

`oflm bench --submit` and `oflm bench-embed --submit` request a short-lived HMAC token from `GET /api/token` and submit validated CSV rows to `POST /api/benchmarks` (matching `src/src/benchmarking.hpp` and `src/src/benchmark_embed.hpp`). The CLI hashes an anonymous per-install UUID (`~/.config/oflm/bench_consent.json` on Linux, `~/.oflm/bench_consent.json` on Windows) into `install_hash` and includes host hardware fields only when the user opted in.

## 4. GitHub Pages Verification and Deployment Workflow

`.github/workflows/docs-pages.yml` validates pushes and pull requests touching `docs/` or the doc check scripts, and deploys `docs/public/` to GitHub Pages on `main`:

1. Runs `utilities/check_embed_bench_docs.py` and `utilities/check_docs_quality.py`.
2. Installs Go `1.24` and Hugo Extended `0.147.9`, verifying both root (`https://openflowlm.org/`) and subpath (`https://example.github.io/OpenFlowLM-Next/`) builds.
3. On `main` pushes to `Atomic-Germ/OpenFlowLM-Next` (or manual `workflow_dispatch`), resolves `--baseURL` dynamically via `actions/configure-pages@v5` (`docs/CNAME` is intentionally omitted so fork previews never redirect to `openflowlm.org`).
