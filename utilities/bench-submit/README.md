# bench-submit — CLI half of community-benchmark submissions

```bash
python3 utilities/bench-submit/submit.py bench_qwen3.6-35B-A3B_20261014.csv \
  --api https://oflm-bench.<account>.workers.dev \
  --model Qwen3.6-35B-A3B --hardware "Radeon 890M (Ryzen AI HX 370)" \
  --backend "NPU (XDNA2)" --quantization Q4_K_M --runs 5
```

What it does: reads the CSV that `oflm bench` / `oflm bench-embed` already writes
(`src/src/benchmarking.hpp`, `src/src/benchmark_embed.hpp`), fills in CPU/RAM/OS
from the local machine (`/proc/cpuinfo`, `wmic`), mints a 10-minute token from
`GET /api/token`, and `POST`s to `/api/benchmarks`. No static secret is needed.

The web-form twin is `docs/benchmarks-submit.md` → `/benchmarks/submit/`.
Server-side validation (power-of-two stages, `min ≤ avg ≤ max`, ranges) lives in
`benchmarks-worker/src/parse.js` and is unit-tested in `benchmarks-worker/test/`.

See `docs/DEPLOY-CLOUDFLARE.md` for the full Pages + Worker + D1 setup.
