# Prefill performance characterization (Phase 21)

Phase 21 is a **blocking** gate: the production prefill path has to be measured,
not assumed. This tool measures the *uninstrumented* production route — the
capture-only CLIs of every earlier phase add readback and are not benchmarks — and
produces the time-budget attribution Phase 21 asks for.

It does not build kernels, change arithmetic, or relax any numerical gate. It
runs an existing CLI, parses the stage breakdown the engine already prints, and
writes a machine-checkable report.

## What is measured

For every prompt length and every schedule, from the CLI's own stderr:

| Field | Source | Meaning |
|---|---|---|
| `resident_s` | `resident after N s` | weight load; once per process, not per request |
| `prefill_ms` | `prefill N tokens: M ms` | the prefill wall time under test |
| `preproc_ms` | `prefill N tokens` − Σ stage totals | text prep / embedding / host work outside the per-token loop |
| `total_ms` | `t_real=...: X ms` | the route's own accounting for the prefill |
| `part0_ms` | same line | host time blocked on the GEMM dispatches |
| `part1_ms` | same line | host time in the non-GEMM half (DeltaNet rule, attention, tail) |
| `route_ms` | same line | MoE routing host stages (0 for these dense qwen35 layers) |
| `lm_head_ms` | same line | the LM head at the last position |
| `mid/dn_conv/dn_rule/attn/prenorm/gemm_tile/gemm_tr/tail/shared/state/sync/setup` | second line | the finer host split |
| `dispatch` per kernel | `NAME calls ms total ...` | per-kernel dispatch time and call count |

Two derived numbers matter:

* `attributed_ms = preproc + total` and the **residual** against `prefill_ms`.
  Phase 21 requires ≥90 % of wall time explained; the residual is what is not.
* `tok_per_s = N / (prefill_ms/1000)`.

The engine reports *host-blocked* time per stage, so `part0 + part1 + route +
lm_head` is the accounting identity inside `total_ms`. NPU busy time is not
directly observable per kernel; `dispatch` sums are the closest available proxy
and are reported as such, never as "NPU utilization".

## Usage

```bash
source utilities/env.sh
export OPENBLAS_NUM_THREADS=1

ironvenv/bin/python utilities/prefill-perf/measure.py run \
  --cli Models/qwen38-27b/post-prefill-production/plain-v2/build/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-v2/kernels-model-ctx2048 \
  --lengths 1,32,128,256,512,2048 \
  --schedules sequential,hybrid,block_major \
  --reps 5 --warmup 1 \
  --out Models/qwen38-27b/prefill-perf/v1
ironvenv/bin/python utilities/prefill-perf/measure.py audit \
  --out Models/qwen38-27b/prefill-perf/v1
```

`run` takes the shared resident-validation lock, requires the NPU to be idle, and
preflights memory, so it never overlaps another NPU job. Prompts are synthetic
token ids (prefill cost depends on the token *count*, not on which ids), which is
what makes a 2048-token prompt possible without a real corpus.

Warm-up runs are executed and discarded: the first run after the NPU idles is
slow (cold hardware context, cold page cache, possible power-mode settle), and a
single run is not a measurement. Each reported row is the **median** of `--reps`
runs with the full spread retained; conclusions are never drawn from one run.

`audit` re-reads the raw per-run logs, recomputes every statistic from them, and
refuses a report whose numbers no longer follow from its logs. Exit 0 means the
measurement is reproducible evidence, **not** that the performance gate passed —
`model_acceptance` stays false.

## Verdict

The report carries `attribution_coverage` per row and an overall
`budget_complete`. A row whose residual exceeds 10 % of wall time is listed under
`unattributed`; Phase 21 requires that list to be explained or explicitly
declared unmeasurable with today's tooling.

## Scope

This is a performance measurement of an existing binary. It makes no numerical
claim, promotes no kernel, and does not touch production arithmetic, catalogue
entries, packing or any installed service. Phase 22 remediation and every
acceptance threshold are separate, and an exception to the GPU comparison is a
human decision that this tool records but never takes.
