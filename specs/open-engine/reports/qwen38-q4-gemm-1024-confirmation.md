# 1024-token prefill confirmation — final performance decision

## Performance verdict: **PASS for the opt-in candidate on this host**

The fixed 14-pair campaign meets the predeclared performance rule. The initial
five-pair campaign remains a failed gate (−0.60%, 4/5 faster); these are separate
experiments, not interchangeable estimates. The criterion in
[`qwen38-q4-gemm-1024-confirmation-criterion.md`](qwen38-q4-gemm-1024-confirmation-criterion.md)
(fixed on disk before any data was collected) is met on every point, and the
conclusion does not depend on the single slow control observation. This is a
single-host result, not a platform-wide performance guarantee. Merge readiness
and evidence limitations are listed in the [acceptance review](qwen38-q4-gemm-acceptance.md).

The production default stays at one DMA visit per TaskGroup. The candidate
remains opt-in and requires explicit approval before any default change.

## Result against the predeclared rule

| Criterion | Threshold | Measured | Pass |
|---|---|---:|---|
| Median relative improvement | ≥ 1.00% | **2.43%** | yes |
| Pairs faster | ≥ 12 of 14 | **14 of 14** | yes |
| Bootstrap 95% CI of median Δ | entirely < 0 | **−3683 … −1984 ms** | yes |
| Decode median change | ≤ +3% | **−0.72%** | yes |
| RSS median change | ≤ +3% | **+0.001%** | yes |

One-sided exact sign test at 14/14: **p = 0.00006**.

## Paired data

14 measured pairs, one warm-up pair excluded, alternating A/B then B/A (each arm
leading seven times). Δ = candidate − control, negative = candidate faster.

Δ (ms): −4914, −3663, −1526, −1092, −3683, −2815, −2612, −4479, −3141, −1984,
−1768, −34047, −2112, −3146.

| Statistic | Value |
|---|---:|
| Median Δ | **−2978 ms** |
| Interquartile range | −3683 … −1984 ms |
| Median relative improvement | **−2.43%** |
| Pairs faster | 14 / 14 |

### The outlier, and why the decision does not rest on it

Pair 11 shows −22.3%: its control run took **152.9 s**, whereas the other
controls took 120.2–123.6 s. The cause of this slow observation is **not
established**. Removing that pair in a sensitivity analysis gives **13/13 pairs faster**,
median **−2.32%**, CI **−3663 … −1984 ms** — the same conclusion, a tighter
interval. The primary result includes all 14 pairs; exclusion is not part of
the acceptance rule.

All 14 candidate runs stayed in **116.5–121.1 s**. Those timings alone do not
identify host interference, a thermal cause or a device bottleneck.

Statistics use Δᵢ = afterᵢ − beforeᵢ and rᵢ = 100 × (afterᵢ / beforeᵢ − 1).
The reported improvement is median(rᵢ), not the ratio of arm medians. Arm medians
are 121,395 → 118,727 ms (−2.20%). The IQR uses medians of the sorted lower/upper
seven deltas. The percentile CI uses 10,000 resamples of 14 paired deltas with
replacement, Python's random generator seeded with 20261115, and sorted indices
249 and 9749 (zero-based). The seed specifies reproduction of the analysis;
the criterion was fixed on disk before collection, not separately committed then.

## Conditions

| Item | Value |
|---|---|
| Host | Ryzen AI 9 365, 91 GiB, NPU Strix (8 columns) |
| Runtime | Benchmark launched through the tree environment; loaded per-process XRT libraries NOT RECORDED. Separate capacity probes verify XRT 2.25, not the benchmark runtime. |
| Driver / firmware | amdxdna 2.26.0_20260817 / 1.1.2.64 |
| Boot | `9285622f-257b-4e83-87b0-494393f16623` |
| Power mode | `default` (unchanged), governor `powersave` |
| Load average at start | 0.92 / 0.86 / 1.44 |
| Thermal | acpitz 54 °C (not NPU die temperature) |
| Context capacity before / after | Three observations of 16 each, no parsed errors in the probe windows; not continuous campaign monitoring |
| Arms | only the 20 GEMM files differ (72 other files byte-identical) |

Thermal and load were not controlled by the experiment; they are recorded so the
result is attributable to this box and session. A different machine may differ.

## Consolidated performance picture

| Tokens | Before | After | Change | Evidence |
|---|---:|---:|---:|---|
| 256 | 29 432 ms | 28 759 ms | −2.29% | 5/5 pairs (earlier run) |
| 512 | 59 365 ms | 58 186 ms | −1.99% | 5/5 pairs (earlier run) |
| 1024, initial | 118 743 ms | 118 029 ms | −0.60% | 4/5 pairs — **FAIL** |
| 1024, confirmation | 121 395 ms | 118 727 ms | −2.20% | Arm-median ratio; **paired median −2.43%, 14/14 faster** |
| 3008 engine prompt tokens | 377 564.5 ms | 369 279.5 ms | −2.19% | Two runs per arm, balanced order; supporting only |

The two 1024-token results are retained independently. Earlier campaigns have
different repetition counts and do not establish identical runtime provenance.
Decode and process RSS stay within the measured gates; zero cost is not proven.

## Merge recommendation

**The performance result supports carrying an opt-in experiment**, not changing
shipped recipes. It is not an unconditional merge approval:

- Numerically byte-identical to the control across 64 layers (147 checks).
- Direct production CLI checks pass on both arms; 9/9 four-layer boundaries match.
- The retained server isolation run uses production kernels, not the candidate.
- Decode and RSS meet the measured non-regression gates.
- The fixed 14-pair 1024-token campaign meets the performance criterion.

Conditions on that recommendation:

1. **Default unchanged.** `GQP_DMA_BATCH` stays 1; no recipe opts in. Promotion
   requires separate approval and target-platform validation.
2. Still **not measured**: candidate server isolation, independent full-layer
   residual oracle, HTTP TTFT, Windows, driver-owned NPU memory, freshly rebuilt
   tiled exports and thermal-controlled re-runs. Per-arm benchmark XRT 2.25
   identity is unverified. See the acceptance review for branch-scope prerequisites.

No additional DMA tuning or benchmark extension is proposed by this report.

## Next optimization target

The [profiling reference](qwen38-npu-performance.md) explains the nested host
counters. GEMM submit/wait accounting remains the largest observed region in
the retained logs; it is not a device-active or bandwidth measurement. The
unassigned `residue` counter is not a named host stage, and nested counters
must not be added as independent shares. A separate profiling task can distinguish
weight transfer, dequantization, compute and waits before choosing another
optimization. No such bottleneck attribution or redesign is established here.

## Evidence

`Models/qwen38-27b/pr3-gemm/confirm-1024/` (result.json, per-arm logs, ids).
Statistics: `confirm-1024-stats.json`. Criterion:
`specs/open-engine/reports/qwen38-q4-gemm-1024-confirmation-criterion.md`.
Capacity: `capacity-confirm-pre-*/`, `capacity-confirm-post-*/`.
Raw result SHA-256: `b7bb4994f5ed6e12b89368b39f50b22713b79ace930199992bdfb823318a1503`.
