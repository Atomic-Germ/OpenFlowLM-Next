# Separate FFN projection validation

`probe.py` exposes FP32 norm and per-band `[up[64], gate[64], h[64]]` captures.
The all-layer sweep uses `FRONT_FP32=2`: direct FP32 table preparation with the
same consistent Q4 activation sums as the production wide dense recipe. Modes
0 (BF16 prep) and 1 (direct FP32 with original sums) remain available for the
historical single-layer experiment in `check.py`.

Build with `front_build.py --spec EXPORT/spec.json --out BUILD`. The builder
selects mode 2, pins source and binary hashes and audits copied compiler sources
and production header dependencies. It rejects unsupported geometry and Q8 FFN.
The probe is a diagnostic artifact; it is not an application kernel export.

Prepare all layers and positions from an accepted full-depth fixture:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/ffn-front-replay/sweep.py \
  --out OUTPUT --model MODEL --kernels EXPORT --fixture FIXTURE \
  --pools POOLS --build BUILD \
  --original-inputs PRODUCTION/input-sha256.json \
  --original-captures PRODUCTION/stage-evidence.json
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  HARNESS OUTPUT/run.cfg
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/ffn-front-replay/sweep.py \
  --out OUTPUT --score
```

One pool buffer is reused across layers. Scoring loads one layer's up/gate
weights at a time; full-model NPU residency is unnecessary. Input provenance
includes the original model, every pool, production captures, source files,
configuration and compiled probe. Missing coverage, changed files, partial
fixtures and overlapping output/source paths are rejected. Preparation clears
old captures; each scoring attempt removes stale success before verification.

Every case must satisfy both:

- Probe norm and combined activation `h` match the production capture bytewise.
- Independent FP64 norm, up, gate and activation references pass the unchanged
  normalized maximum-error bound `<0.005`. The reference never substitutes a
  captured or BF16-rounded norm. A correct product cannot hide a failed projection.

These are isolated replay measurements on captured pre-FFN residuals. Keep
them alongside full-depth residual/logit checks; they do not establish prefill,
source-model conversion parity, serving isolation or performance acceptance.

The [runbook](../../.agents/skills/open-qwen38-ffn-projection-sweep/SKILL.md)
records the native toolchain, complete commands, resource limits and measurements.
