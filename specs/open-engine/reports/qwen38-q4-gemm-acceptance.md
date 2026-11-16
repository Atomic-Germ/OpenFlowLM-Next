# PR #3 acceptance — bounded Q4 GEMM DMA batching

## Review verdict: **opt-in performance passes; upstream merge is conditional**

The fixed [14-pair confirmation](qwen38-q4-gemm-1024-confirmation.md) passes its
predeclared performance rule: median paired prefill change −2.43%, 14/14 faster,
95% bootstrap CI for median Δ −3683 to −1984 ms. The initial five-pair 1024-token
experiment remains FAIL. Full-depth captured logits and states match exactly.
These results support the opt-in implementation, not production promotion or
an assertion that every requested integration/runtime gate has been measured.

Nothing here enables the candidate, changes a numerical tolerance, changes
model precision, or alters the default DMA setting.

## Base and dependency status

**PR #3 is stacked on PR #2; PR #2 has not landed upstream.**

- Intended PR #3 base: `836072db` (`perf/npu-prefill-scheduling` tip, "docs:
  record merged Qwen3.8 prefill acceptance").
- Actual upstream main at review: `9b1409d3`.
- `git merge-base --is-ancestor 836072db upstream/main` exits **1**, and neither
  `4573739c` (OpenMP isolation), `1448fc65` (DeltaNet head mapping),
  `78c41524`/`9f8415c9` nor `836072db` is an ancestor of upstream main.
- `836072db..HEAD` contains only PR #3 commits (the DMA design, diagnostics,
  evidence and review). `upstream/main..HEAD` also contains PR #2
  (`004821d0`, `8ba65ae9`, `78c41524`, `9f8415c9`, `4573739c`, `1448fc65`,
  `836072db`), merges of upstream main and earlier PR #1 profiler work.
- The upstream diff is therefore much larger than the PR #3 diff. The checked-in
  JSON records the exact commit and file counts; `git diff --shortstat` against
  each base is authoritative and moves with each review commit.

Review against `836072db` (Option B), or land PR #2 first and re-target. A
rebase onto upstream main was **not** performed: it would either absorb PR #2
changes into this PR or require rewriting history, neither authorized here.
Do not present the 65-file diff as a standalone GEMM-only PR.

Within the PR #3 slice, production changes are limited to the DMA planner/design,
export-environment isolation and copying/hashing the planner in five existing
replay builders. Benchmark-driver extensions, fixtures, tests and evidence serve
that experiment. No runtime C++, weights, compiled artifacts, registry entries,
recipes, tolerances, attention or DeltaNet arithmetic change in this slice.
The context-capacity utility/report are the largest auxiliary changes; they
address the explicitly requested blocker investigation and remain separate
diagnostic commits. They are not needed to enable batching and can be reviewed
separately rather than folded into the optimization's rationale.

## Candidate server validation — PASS

A new run (`Models/qwen38-27b/pr3-gemm/accept-server-runtime/`) pointed
`OFLM_OPEN_KERNELS_DIR` at each private arm and drove the real `oflm serve`.

| Check | Control `model-batch1` | Candidate `model-batch4` |
|---|---|---|
| Engine selection log | arm path, `OFLM_OPEN_KERNELS_DIR` | arm path, `OFLM_OPEN_KERNELS_DIR` |
| GEMM artifacts | 10 programs, manifest `744879192c49…` | 10 programs, same manifest |
| 64-layer model init | PASS | PASS |
| A, B then A again | all answered | all answered |
| A before B == A after B | byte-identical | byte-identical |
| B == fresh-process B | identical | identical |
| A ≠ B | distinct prompts | distinct prompts |
| Runtime mappings | XRT 2.25.00 | XRT 2.25.00 |
| Kernel errors / capacity | none parsed; no contexts left resident | none parsed; no contexts left resident |
| Result | **PASS 11/11 graded**, reset N/A | **PASS 11/11 graded**, reset N/A |

All 20 GEMM binaries differ between arms (`insts.bin` and `final.xclbin`);
the manifest is byte-identical. Control and candidate return the same A and B
answers, so the candidate does not change server output. `/v1/context/reset`
and the other reset paths still 404; recorded as N/A, not a pass.

## Runtime provenance

The historical benchmark processes preserved no `/proc/<pid>/maps`. The claim
that they ran on XRT 2.25 was **wrong**: re-running the exact 64-layer benchmark
command shape under `utilities/env.sh` loads
`/opt/xilinx/xrt/lib/libxrt_{core,coreutil,driver_xdna}.so.2.26.0` for **both**
arms (`runtime-bench-batch{1,4}/provenance.json`). `ofllm` links the distro
soname, but `env.sh` prepends `/opt`, so the loader chooses 2.26.

The server run above deliberately avoids `env.sh` and the `oflm_drive` helper's
library path, so it maps `/usr/lib/x86_64-linux-gnu/libxrt_*.so.2.25.00` for
both arms.

Consequence: the historical 14-pair comparison remains internally valid, since
both arms used the same 2.26 runtime and the same binary; it is a **2.26
result**, not a 2.25 result. No measurement is reinterpreted or overwritten.
`utilities/prefill-block/omp_policy.py` now samples each timed process's maps
and stores `runtime_libraries` per row, so future campaigns carry this proof
without adding tracing to the inference window.

Remaining material gap: candidate and control were not re-timed end-to-end on
XRT 2.25. The opt-in candidate's server path is validated on 2.25; its 14-pair
prefill number is a 2.26 number. Re-timing is a bounded future confirmation,
not needed to accept the implementation, and it was not run here.

The performance arm is complete; no more DMA tuning or 64-layer rerun is
requested by this review. No history rewrite, default promotion, merge or push
is performed here.

## Host verification

The portable suite passes **2,757 tests, with 172 skips**; skipped capabilities
are not passes. The focused review suite passes 212 tests. The retained kernel
audit verifies 1,955 files and 1,100 dispatch records. A read-only confirmation
audit reproduces all 30 process records (one warm-up pair plus 14 measured pairs),
the paired statistics, 187 binary/manifest provenance paths and the 20 changed /
72 unchanged kernel-file separation. The 19.8 GB model hash is retained from the
campaign, not re-read in this review. Settings scratch-merge/regeneration passes
against upstream main `9b1409d3`; whitespace checks pass. No new hardware
execution, kernel build or benchmark campaign is performed for this review.

## What was compared

| Item | Before | After |
|---|---|---|
| Kernel set | `model-batch1` (`GQP_DMA_BATCH=1`) | `model-batch4` (`GQP_DMA_BATCH=4`) |
| Manifest | identical (`744879192c49…`) | identical |
| Binary | `open_qwen36_cli` / `oflm` same build | same build |
| Runtime | XRT 2.26.0 under `utilities/env.sh` (reproduced) | XRT 2.26.0 under the same shell |
| Device | NPU Strix, 8 columns | same |

Arm separation verified: 92 kernel files compared, **only the 20 GEMM
`final.xclbin`/`insts.bin` files differ**; all 72 non-GEMM files are
byte-identical. The experiment therefore isolates the DMA schedule.

Commits: baseline `836072db`; candidate `34419a6b` (opt-in DMA batching),
diagnostic record `b4ee9f22`.

## Numerical correctness — PASS

Full-depth comparison, 64 layers, 256-token prompt, 16-token continuation
(`accept-full64/`):

| Check | Result |
|---|---|
| Total checks | **147, zero FAIL** |
| Per-layer DeltaNet conv + state | 48 linear layers |
| Per-layer attention KV | 16 layers |
| Logits rows (prompt + 16 decode) | 17, cosine 1.0, max_abs 0.0 |
| Greedy continuation | identical 16 IDs |
| Logits bytes | **byte-exact** |
| State slices | **byte-exact** |

The 112 state slices cover 48 linear layers (conv + recurrent state) and
16 attention layers (KV). Existing comparison thresholds are unchanged.
This is control/candidate equivalence for the captured boundaries, not an
independent per-layer residual-oracle measurement; that gate is NOT MEASURED.

## Prefill performance

The 256/512 and initial 1024 experiments use five alternating pairs each and one
excluded warm-up pair. The confirmation uses exactly 14 measured pairs and one
excluded warm-up pair. All use the same binary and benchmark-driver kernel mode.

| Tokens | Before median | After median | Change | Pairs faster | Verdict |
|---|---:|---:|---:|---:|---|
| 256 | 29 432 ms | 28 759 ms | **−2.29%** | 5/5 | PASS |
| 512 | 59 365 ms | 58 186 ms | **−1.99%** | 5/5 | PASS |
| 1024, initial | 118 743 ms | 118 029 ms | **−0.60%** | 4/5 | **FAIL** |
| 1024, confirmation | 121 395 ms | 118 727 ms | −2.20% arm-median ratio; **−2.43% paired median** | 14/14 | **PASS** |
| 3008 engine tokens | 377 564.5 ms | 369 279.5 ms | **−2.19%** | Two runs per arm | Supporting only |

- 256 and 512 pass the project's paired gate (all five pairs faster; one-sided
  sign test p = 1/32).
- **1024 fails**: only four of five pairs improved, and the −0.60% median
  (711 ms) is inside the observed run-to-run spread (before 118.3–121.1 s,
  after 116.8–119.6 s). This is a genuine gate failure, not an infrastructure
  error: every process exited 0 and produced identical output.
- The longer-prompt point uses 3000 text tokens, **3008 engine prompt tokens**,
  a 4096 context allocation, 16 generated tokens and order control/candidate/
  candidate/control. It is not a 4096-token prompt or a five-pair result.
- A secondary diagnostic that ran both longer-prompt arms against the **same** default
  kernel set was discarded; only the runs whose recorded `kernels` path differs
  per arm are reported.

Prefill throughput at 256 tokens is 8.70 vs 8.91 tok/s; at 512 tokens 8.62 vs
8.80 tok/s.

## Decode — no regression

| Scenario | Before | After | Change |
|---|---:|---:|---|
| 256-token prompt | 460 ms/token | 459 ms/token | −0.22% |
| 512-token prompt | 468 ms/token | 471 ms/token | +0.64% |
| 1024-token prompt | 475 ms/token | 478 ms/token | +0.63% |
| 3008-token engine prompt, context allocation 4096 | 1.6525 tok/s | 1.647 tok/s | −0.33% |
| 1024-token confirmation (14 pairs) | 485 ms/token | 481.5 ms/token | −0.72% |

All within the pre-existing ±3% decode gate on this host. HTTP streaming TTFT
is NOT MEASURED; the reported prefill window excludes model loading.

## Memory — unchanged

Peak process RSS did not move: 34 278 168 → 34 278 116 KiB at 256 (−0.00%),
34 277 296 → 34 277 208 at 512, 34 277 432 → 34 277 664 at 1024. Driver-owned
NPU memory was **NOT MEASURED** (no supported counter on this stack).

## Integration — captured coverage

| Gate | Result |
|---|---|
| `oflm run`, both arms | exit 0, 64/64 layers, identical answer `RED` |
| `oflm serve`, control arm A→B→A | 11/11 graded PASS, arm selection logged |
| `oflm serve`, candidate arm A→B→A | 11/11 graded PASS, arm selection logged |
| A before B equals A after B | PASS, byte-identical in both arms |
| B matches a fresh-process B | PASS in both arms |
| `/v1/context/reset` etc. | N/A — no reset endpoint on this build |
| Boundaries 95/96/97 | 3/3 PASS, exact bytes |
| Boundaries 255/256/257 | 3/3 PASS, exact bytes |
| Boundaries 511/512/513 | 3/3 PASS, exact bytes |

Boundary runs used 4 layers and 2 decode tokens for speed; full-depth coverage
comes from the 64-layer run above.

`oflm` needed `src/lib/xrt` on `LD_LIBRARY_PATH` in this checkout because the
binary carries no RUNPATH to the closed engine libraries. That is an
environment/packaging condition, not a candidate behaviour, and it applied
equally to both arms.

## Context capacity — did not recur

Selected phase-boundary probes (not continuous observation):

| Point | Capacity | Kernel errors |
|---|---|---|
| Start of acceptance | 16, 16, 16 | 0 |
| After all phases | 16, 16, 16 | 0 |

Same boot (`9285622f…`). The 16→8 transition observed earlier in this
investigation did not reproduce in these observations. It remains an
**unresolved intermittent runtime issue**; attribution to firmware or GEMM
is not established. A separate fresh-boot
diagnostic run also completed without a decrease (`b4ee9f22`).

## Evidence

`Models/qwen38-27b/pr3-gemm/`: `accept-full64/`, `e2e-256/`,
`accept-e2e-512/`, `accept-e2e-1024/`, `accept-decode4k-valid.json`,
`accept-run-batch{1,4}.log`, `accept-boundary-*/`, `capacity-phase2-*/`,
`capacity-postphase-*/`.

Candidate server: `accept-server-runtime/` (`result.json`, per-arm
`result.json`, `artifacts.json`, server logs, live library hashes), with
`utilities/prefill-block/server_candidate_check.py`.

Runtime provenance: `runtime-bench-batch{1,4}/provenance.json` for the benchmark
command shape, `accept-runtime-envsh/provenance.json` for the `env.sh` shell.
The older production-kernel server result remains at
`utilities/prefill-block/runs/isolation-aba-20261115-223324/result.json` and is
superseded for candidate validation.

Not measured: HTTP TTFT, independent per-layer residual oracle, Windows,
driver-owned NPU memory, freshly rebuilt tiled exports, context/numerical
capability beyond the recorded prompt lengths, and a 14-pair re-timing on XRT
2.25 (the retained performance run is XRT 2.26 for both arms).
