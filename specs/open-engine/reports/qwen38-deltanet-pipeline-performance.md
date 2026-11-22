# Qwen3.8-27B DeltaNet and CPU/NPU pipeline: profiling report (PR #4)

Profiling only. No production source, kernel, recipe, default, precision or
scheduling change is in this branch. Machine-readable evidence:
[qwen38-deltanet-pipeline-performance.json](qwen38-deltanet-pipeline-performance.json).

## 1. Baseline and runtime

| item | value |
|---|---|
| branch | `perf/npu-deltanet-pipeline` (created from the tree at `9d77431a`) |
| baseline commit | **`9d77431a683ffafc4060c59debe463d696670f4a`** (fork `main`; upstream `main` `d7c145f5` is its merge base for `src/`) |
| excluded | PR #3B/3D experiments (activation replay, dequant variants); they are preserved on their own branches and are absent here |
| model | `Models/qwen38-27b/Qwen3.8-27B-NPU2` (`model.q4nx`, 19,150,596,832 B) |
| kernels | `Models/qwen38-27b/production-v2/kernels-model` (manifest spec hash recorded in the JSON) |
| CLI | built from this branch by `utilities/prefill-block/build-cli.sh`; sha256 in the JSON |
| precision / route | paired prefill, block loop (`OFLM_OPEN_LAYER_MAJOR=0`), decode route `npu` |
| runtime | XRT **2.26.0** `libxrt_core`, `libxrt_coreutil`, `libxrt_driver_xdna` from `/opt/xilinx/xrt/lib`, hashes captured from the live process |
| driver / firmware | amdxdna `2.26.0_20260817` (`2b71d02b…`), firmware `1.1.2.64` (previously recorded) |
| machine | Ryzen AI 9 365, 91 GiB RAM, XDNA2 Strix, 8 columns, Ubuntu 26.04.1, Linux 7.0.0-43-generic |
| power | NPU power mode `default` (unchanged), CPU governor `powersave` (unchanged) |

Measured 2026-11-22 (UTC). No NPU or compiler workload ran concurrently with a
timed capture. This desktop is not a controlled-latency laboratory; load and
thermal snapshots are recorded per capture.

`oflm` was not used because the model directory has no `open_kernels` link; the
profiler's own CLI drives the same engine, weights and kernel set.

## 2. Measured breakdown

### 2.1 Prefill, 256 tokens (production block route)

Two independent captures of the same request; both pass the profiler's three
accounting checks (layer walls + lm_head inside the request, named host stages
inside `host_ms`, DeltaNet conv+rule inside `mid_ms`).

| column | capture A | capture B | share of request (A) |
|---|---:|---:|---:|
| request total | 29,243.1 ms | 29,002.6 ms | 100% |
| GEMM dispatches (`part0`, submit-to-completion) | 24,580.0 ms | 24,326.6 ms | 84.1% |
| host stages (`part1`) | 1,912.1 ms | 1,912.4 ms | 6.5% |
| **not in part0/part1** | **2,725.2 ms** | **2,738.0 ms** | **9.3%** |
| lm_head (outside layer walls) | 25.9 ms | 27.6 ms | 0.1% |

Host-stage detail (A): `gemm_tile` 463.6, `gemm_tr` 316.5, `sync` 290.0,
`mid` 786.2 (of which `attn` 290.5, **`dn_conv` 217.8**, **`dn_rule` 277.8**),
`shared` 147.1, `tail` 36.4, `prenorm` 19.3, `state` 4.9.

Per layer kind: 48 linear layers 451.6 ms each (GEMM 387.3, host 27.9),
16 full layers 456.0 ms each (GEMM 372.4, host 35.7). **DeltaNet layers are not
disproportionate**; the full-attention layers cost 4.4 ms more each.

### 2.2 What the 2.7 s is

The profiler's "unattributed to stages" column is
`request − lm_head − part0 − part1`. It is **not** unmeasured CPU code inside
the layer body. Reading `core.cpp`/`prefill_precision.hpp`:

- `part0` is `run()` = submit + blocking `wait_run()`, and `wait_run` books only
  `xrt::run::wait` into `part1`. So a blocking GEMM dispatch's **device time is
  in `part0`, and any host work between two dispatches falls in neither column.**
- The paired-prefill fold runs exactly there. `prefill_precision::Workspace::run`
  splits the input (`K·T`), copies the high product (`N·T`), adds the low input
  product (`N·T`), then `InputWorkspace::run` splits the **same** input again and
  the outer `run` copies and accumulates once more — all on the CPU, strictly
  between blocking dispatches.
- Two tagged windows also sit outside `part1` by construction: `shared_ms`
  (147.1 ms, silu·up/accumulate) and `state_ms` (4.9 ms). Subtracting them,
  **2,573 ms of the 2,725 ms is the precision fold and other untagged folds.**

Independent confirmation: a probe compiled against the **real**
`prefill_precision.hpp` with a work-free dispatch, on the same 64-layer shape
(48 linear + 16 full, T=256), measures the fold alone at **3,944.7 ms**
(minimum of 7; 4,490.9 ms first run). The in-engine figure is lower because the
dispatch outputs are already-mapped BOs and some copies land in cache, but the
magnitude and the serial placement agree.

### 2.3 Decode

Sequential route, 3 profiled steps after a 32-token prompt: **1,264 ms for 3
tokens** (about 421 ms/token), of which `part0` is 4 ms and `part1` 1,258 ms.
`dispatch_ms` is 650 ms per step against 424 ms of wall — the intervals
**overlap** under the decode pipeline, so this column is queue residence and not
CPU work.

Process-level CPU time makes the split explicit:

| capture | wall | CPU time | CPU fraction |
|---|---:|---:|---:|
| 256-token prefill block | 35.50 s | 21.32 s | 0.60 |
| 32-token prompt + 8 decoded tokens | 23.52 s | 6.86 s | 0.29 |

Decode is predominantly **NPU wait**, and it is already pipelined
(`OPEN-DECODE-PIPELINE`, submit-ahead level 1, one context). There is no
DeltaNet-specific decode cost visible in the columns: `dn_conv`/`dn_rule`/`mid`
are ~0 because the fused per-layer `lx` dispatch performs the recurrence on the
device on that route.

## 3. Critical-path analysis

The block route is a chain of **blocking** GEMM dispatches: `run()` starts a
command and immediately waits. The precision fold, tiling, transposes, DeltaNet
conv/rule and state syncs all execute on the CPU **between** those waits, so
their time is additive to prefill latency; none of it overlaps the NPU. That is
why `wall ≈ part0 + part1 + untagged` holds to 0.1% in both captures.

Ranked by additive critical-path cost per 256-token block:

| suboperation | ms | % of request | notes |
|---|---:|---:|---|
| paired-prefill host fold (split/copy/accumulate) | ~2,573 | ~8.8% | 64 layers × 4 projections |
| GEMM output tiling (`gemm_tile`) | 463.6 | 1.6% | AVX2, writes mapped BO |
| GEMM output transpose (`gemm_tr`) | 316.5 | 1.1% | AVX2 |
| BO host↔device syncs (`sync`) | 290.0 | 1.0% | `x` up / `y` down per dispatch |
| DeltaNet rule on S (`dn_rule`) | 277.8 | 1.0% | 48 layers, 128 threads of work |
| DeltaNet conv + q/k norm + α/β (`dn_conv`) | 217.8 | 0.7% | 48 layers |
| full-attention host half (`attn`) | 290.5 | 1.0% | 16 layers |
| fused FFN silu·up/accumulate (`shared`) | 147.1 | 0.5% | |
| residuals / post-norm / router (`tail`) | 36.4 | 0.1% | |
| input RMSNorm (`prenorm`) | 19.3 | 0.1% | |
| DeltaNet state syncs (`state`) | 4.9 | 0.02% | |

**DeltaNet is not the bottleneck.** Conv + rule + its state syncs total
**500.5 ms, 1.7%** of the request, and both phases are already parallelised
(tokens for the conv, value heads for the rule).

## 4. Ranked optimization candidates

| # | candidate | est. saved per 256-token block | % of prefill | complexity | numerical risk | sync/memory risk |
|---|---|---:|---:|---|---|---|
| 1 | **Host fold de-duplication** (one input split, no redundant intermediate copy) | **~1.47 s** (measured ceiling, 57% of the fold) | **~5.0%** | low–moderate | low: association `(A+B)+(C+D)` kept elementwise | none: no dispatch, context or buffer-size change |
| 2 | Fuse `gemm_tr` transpose into the consumer (DeltaNet conv reads `[N,T]` directly) | ≤0.32 s | ≤1.1% | high | medium: indexing rewrite across consumer kernels | none |
| 3 | Overlap the fold with the next dispatch (split GEMM into submit-ahead phases) | up to ~2.6 s on paper | ≤8.8% | very high | low | high: reorders host/device access to shared BOs; the engine already documents why cross-context queue-ahead is unsafe |
| 4 | Tune `gemm_tile`/`sync` (prefetch, non-temporal stores, fewer syncs) | ≤0.75 s | ≤2.6% | moderate–high | low | medium: BO mapping lifetime |
| 5 | DeltaNet conv/rule micro-tuning | ≤0.25 s | ≤0.9% | moderate | medium (FP32 state, causal order) | none |
| 6 | Move the DeltaNet recurrence to the NPU | negative to ~0.5 s | ≤1.7% | very high | high | adds dispatches; the plan forbids assuming a win |

Candidate 1 is the only one that is simultaneously **measurable, arithmetic
preserving, dispatch-free and isolated to one header**. Candidate 3 has the
largest theoretical ceiling but conflicts with the engine's documented
submit-ahead safety invariant and is disproportionate for this PR.

### Measured ceiling for candidate 1

Probe: the real `prefill_precision.hpp` baseline versus a bit-identical
re-association (`((A+B) + (C+D))` in both), T=256, 48 linear + 16 full layers,
minimum of 7 runs.

| variant | ms |
|---|---:|
| baseline: two input splits + redundant intermediate copy | 3,432.8 |
| candidate: one split, no redundant copy | 1,477.4 |
| **ceiling saved** | **1,955.4 (57.0% of the fold)** |

Applied to the observed untagged fold (2,573 ms), that ceiling is **≈1.47 s per
256-token block**, i.e. **≈5.0% of the 29.0 s prefill request** and essentially
the whole TTFT improvement available from host-side work at this shape. A
256-token prefill would move from ~29.0 s to ~27.5 s if the engine realises the
ceiling; the real figure must be measured, not assumed.

## 5. Decision

**GO — narrowly.** A real, serial, arithmetic-preserving host cost of ~2.6 s per
prefill block is on the critical path, and a measured ceiling of ~1.47 s exists
without touching kernels, precision, dispatch count or context count. DeltaNet
is *not* the target; the precision fold is.

The decision is **not** a general GO on "DeltaNet optimization": conv/rule
together are 1.7% of the block and are already parallel. Candidates 2–4 are
recorded but not recommended for the next phase.

## 6. Proposed PR #4B scope (awaiting review)

`perf/npu-prefill-fold`:

1. Rewrite `prefill_precision::Workspace::run`/`InputWorkspace::run` to split the
   input once per projection and to drop one intermediate `N·T` copy, keeping the
   per-element association `((W_hiX_hi + W_hiX_lo) + (W_loX_hi + W_loX_lo))`
   and FP32 accumulation unchanged.
2. No kernel, recipe, manifest, quantisation, dispatch-count, context-count or
   default change; the fold stays on the host.
3. Gates: byte-identical outputs against the current binary on the existing
   prefill precision fixtures; the existing paired-vs-sequential oracle
   (`cosine ≥ 0.9999`, unchanged); 256/512/1024-token prefill and short + ~4K
   decode with matched arms; `OFLM_OPEN_PROFILE` off for the timed arms.
4. Expected: ~5% prefill/TTFT improvement if the ceiling holds; **no decode
   claim** (decode does not run the fold and is NPU-wait bound).
5. Stop and report if the measured engine gain is below 2%.

## 7. Verification and limitations

- Both prefill captures pass `layer_profile.py` validation (64 ordered layers,
  complete requests, three accounting checks); the decode capture passes the
  same checks.
- Runtime libraries were sampled from the live benchmark process; hashes and
  paths are in the JSON.
- **NOT MEASURED**: end-to-end TTFT through `oflm run`/`oflm serve`; ~4K-context
  decode; 512/1024-token blocks; other model families; native Windows. The
  2.75 s figure in the task description matches the profiler's *stage-unaccounted*
  column, not a measured CPU-only time; this report replaces it with the
  `part0`/`part1` decomposition and process CPU accounting.
- The ceiling probe excludes BO mapping, cache warmth and dispatch interference;
  it bounds host cost, not engine gain.
- `part0` is host-observed submit-to-completion and includes queue residence; it
  is not device-only execution time, and no NPU utilisation claim is made.
- The profiling branch adds only diagnostics and this report. No production
  default was changed, and nothing was pushed or merged.
