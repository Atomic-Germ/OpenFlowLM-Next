# Qwen3.8-27B DeltaNet and CPU/NPU pipeline (PR #4, corrected by PR #4A)

Profiling and measurement repair only. No production optimisation, kernel,
recipe, precision, threshold or default is changed. Machine-readable evidence:
[qwen38-deltanet-pipeline-performance.json](qwen38-deltanet-pipeline-performance.json).

## Decision

**NEEDS MORE EVIDENCE.** The previous GO rested on a host-fold microbenchmark
whose two arms computed different values (6.0 versus 10.0) and on a decode
profile of prompt tokens. Both are repaired. The corrected evidence shows a
real, directly measured host cost, but not a trustworthy end-to-end ceiling:

* the production fold is **2,522.5 ms** per 256-token block (exclusive measured
  windows: split 1,263.6, copy 610.1, accumulate 648.9), 8.6% of the
  **29,481 ms** median prefill;
* the corrected cache-hot microbenchmark removes **21.0%** of that operation
  (436.5 ms median paired), not the withdrawn 57%;
* applying a cache-hot microbenchmark saving to a production cost dominated by
  mapped-buffer reads is not justified without an engine-level trial.

That is enough to justify a small, gated experiment, not enough to promise a
prefill or TTFT gain. PR #4B should not start until the review accepts this
scope; if it does, it must be time-boxed and stop if the engine gain is absent.

## Withdrawn claims

The earlier version of this report is superseded. **Withdrawn:** the 57%
host-fold improvement, the 1.47 s saving, the 5% end-to-end estimate and the
quantitative GO justification. The old decode capture is retained and marked
**INVALID FOR DECODE ANALYSIS** in the JSON: its three profiled steps were
sequential prompt steps (`lmhead_ms = 0`), not generated tokens.

## 1. Baseline and runtime

| item | value |
|---|---|
| branch | `perf/npu-deltanet-pipeline` |
| baseline | `9d77431a683ffafc4060c59debe463d696670f4a` (upstream `d7c145f5`) |
| excluded | activation-replay (`f207e675`) and dequant (`dc46bdad`) experiments |
| model | `Models/qwen38-27b/Qwen3.8-27B-NPU2` (`model.q4nx` sha256 in JSON) |
| kernels | `Models/qwen38-27b/production-v2/kernels-model` (manifest sha256 in JSON) |
| CLI | `Models/qwen38-27b/pr4a/cli/open_qwen36_cli`, sha256 in JSON |
| route / precision | paired prefill, block loop; decode route `npu` |
| runtime | XRT 2.26.0 core/coreutil/xdna, paths and hashes sampled **from each measured process** |
| machine | Ryzen AI 9 365, XDNA2 Strix 8 columns, Ubuntu 26.04.1, kernel from the capture |
| power | NPU `default`, CPU governor `powersave` (unchanged) |

## 2. Host fold: corrected benchmark and direct production measurement

### 2.1 Numerical equivalence (was broken)

The shared probe (`utilities/prefill-block/host_fold_probe.hpp`) feeds **both**
arms the same four products `a,b,c,d`, applies the production validation, and
compares complete buffers. `compare()` reports acceptance separately, so "both
reject NaN" cannot masquerade as a numeric result. Both arms keep persistent
workspaces, as `Core` keeps `precision_workspaces_`.

Behavioural tests compile and run the probe: random and rounding fixtures must
be byte-identical; NaN must be rejected by both; a deliberately corrupted arm
must fail. These fail on the old implementation.

### 2.2 Fair, warmed, paired benchmark

The benchmark validates every projection before timing, discards one warm-up
pair, alternates arm order, retains each sample and bootstraps the paired median.
T=256, 48 linear + 16 full layers, all four projections each.

| statistic | value |
|---|---:|
| baseline median | 2,070.9 ms |
| candidate median | 1,644.5 ms |
| paired median difference | −436.5 ms |
| paired median change | −21.01% |
| wins | 5/5 |
| bootstrap 95% CI of median difference | [−456.3, −295.1] ms |
| warm-up pairs excluded | 1 |

The candidate removes the duplicate input split and one intermediate copy while
keeping `(A+B)+(C+D)` per element.

### 2.3 Directly measured production fold

`OFLM_OPEN_FOLD_PROFILE` adds exclusive, non-overlapping windows inside
`prefill_precision`; dispatch time is outside every window. It is off unless set.

| phase | ms / 256-token block | share of fold |
|---|---:|---:|
| input split (both, one is redundant) | 1,263.6 | 50.1% |
| intermediate copy (+ final copy) | 610.1 | 24.2% |
| accumulate | 648.9 | 25.7% |
| **total, 256 folds** | **2,522.5** | 100% |

Context: uninstrumented 256-token prefill is **29,481 ms** median of five runs
(29,404–29,544); the fold is therefore **8.6%** of prefill wall. Instrumentation
did not materially inflate this run (29,390 ms), so the fold figure is usable;
even so it is one instrumented run and is labelled single-run.

Note the microbenchmark totals (~2.07 s) are below the production split+copy
alone (1.87 s): production reads mapped buffers and cache-cold memory, so its
cost is not the cache-hot arithmetic the probe measures. The **21%** applies to
the probe's own baseline, not to the 2,522.5 ms.

## 3. Prefill

Five fresh 256-token captures, no fold instrumentation, no profile overhead in
the timed path beyond the optional layer profile:

| statistic | value |
|---|---:|
| median prompt wall | 29,481 ms |
| range | 29,404–29,544 ms |
| model load (separate) | ~6.5 s |
| real prompt tokens | 256 |
| physical block | 256 |

One capture's layer accounting (single run) gives dispatch 24.5 s, host stages
~1.9 s, DeltaNet conv 226.0 ms + rule 280.0 ms + state 4.7 ms. **DeltaNet is
510.7 ms, 1.7%** of the block: not the bottleneck. Linear layers average
6.5 ms more wall each than full layers in capture A (458.1 vs 451.6), so the
earlier "full layers are more expensive" statement was wrong and is corrected.

## 4. Decode (genuine generated tokens)

Captured with `OFLM_OPEN_PROFILE_STEPS=decode:N`, so prompt steps cannot consume
the budget. Seven generated steps at positions 32–38, every one with
`lmhead_ms ≈ 25.5 ms`:

| statistic | value |
|---|---:|
| generated tokens | 7 |
| generated latency | **452.0 ms/token** |
| prompt (32 tokens) | 13,743 ms |
| model load | 6.4 s |
| per-step total | 444.7–451.4 ms |
| per-step `dispatch_ms` | 668.8–678.3 ms (overlaps wall under submit-ahead) |

Decode is dominated by waiting, not CPU: `dispatch_ms` exceeds step wall, so the
intervals are pipelined. Aggregate process CPU over load+prompt+generation is
**not** a decode CPU fraction and is not reported as one; a decode-only CPU/wait
split remains **NOT MEASURED**.

## 5. Ranked candidates (recomputed)

| # | candidate | evidence | expected engine gain | confidence |
|---|---|---|---|---|
| 1 | Host fold de-duplication | 2,522.5 ms fold; 21.0% cache-hot probe gain | 0.4–1.8% of prefill if a quarter to all of the cache-hot share transfers | low–moderate |
| 2 | Tiling/transpose/sync tuning | 1.9 s attributed host stages | ≤2.6% | low |
| 3 | Fuse GEMM transpose into consumers | 316–330 ms | ≤1.1% | low, high complexity |
| 4 | DeltaNet conv/rule tuning | 510.7 ms | ≤1.7% | low, already parallel |
| 5 | Overlap fold with dispatches | — | theoretical only | rejected: reorders shared BO access, engine documents the hazard |

## 6. Why NEEDS MORE EVIDENCE

A directly measured 2.5 s cost exists, so the target is real. But the only
reliable saving measured is on a cache-hot model of the operation; the
production split+copy alone is larger than the entire probe, and no corrected
engine-level A/B of the rewrite has been run. Promising a percentage now would
repeat the error this repair exists to fix.

**Proposed, if approved:** a time-boxed PR #4B that implements only the fold
de-duplication, gates on byte-identical prefill outputs and the existing
`cosine ≥ 0.9999` oracle, and measures 5 alternating prefill pairs at 256 tokens
plus decode preservation. Stop and report if the median engine gain is < 1%.

## 7. Verification and limitations

* Behavioural probe tests, five-pair prefill, genuine decode, settings and
  whitespace checks are recorded in the JSON and re-run by the spec suite.
* Only 256-token prefill and 32+8 decode were measured; 512/1024, ~4K decode,
  TTFT through `oflm run`/`serve`, other families and Windows are **NOT
  MEASURED**.
* `part0`/`dispatch_ms` are host-observed and include queue residence; no
  NPU-utilisation claim is made.
* Fold instrumentation is opt-in and absent from timed arms; its single run is
  labelled as such.
* Production kernels, thresholds and defaults are unchanged; nothing was pushed.
