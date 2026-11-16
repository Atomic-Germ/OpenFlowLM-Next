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

## Scope and remaining merge conditions

Against the intended PR #2 tip `836072db`, the reviewed PR #3 adds 31 files
(4,432 insertions, 31 deletions before this documentation review). Against actual
upstream main `9b1409d3`, the branch includes 65 changed files, including earlier
OpenMP startup-policy and DeltaNet manifest-mapping work. Those are dependencies,
not GEMM changes: land the prerequisite work first or use the intended stacked
base. Do not present the 65-file diff as a standalone GEMM-only PR.

Within the PR #3 slice, production changes are limited to the DMA planner/design,
export-environment isolation and copying/hashing the planner in five existing
replay builders. Benchmark-driver extensions, fixtures, tests and evidence serve
that experiment. No runtime C++, weights, compiled artifacts, registry entries,
recipes, tolerances, attention or DeltaNet arithmetic change in this slice.
The context-capacity utility/report are the largest auxiliary changes; they
address the explicitly requested blocker investigation and remain separate
diagnostic commits. They are not needed to enable batching and can be reviewed
separately rather than folded into the optimization's rationale.

Two evidence qualifications remain material to the requested acceptance matrix:

- The server A→B→A result records the retained production kernel path, not either
  private DMA arm. Candidate server isolation is **NOT MEASURED** by that result.
- The acceptance/confirmation benchmark driver records commands and artifact
  hashes, but not loaded XRT libraries. Separate diagnostic processes verify
  XRT 2.25; they do not prove the benchmark processes used it. The requested
  verified-XRT-2.25 benchmark condition is **UNVERIFIED**, not PASS.

The performance arm is complete; no more DMA tuning or 64-layer rerun is
requested by this review. Keep the upstream PR draft/conditional until its base
is scoped correctly and these qualifications are either satisfied or explicitly
accepted as limitations of an experimental opt-in merge. No history rewrite,
default promotion, merge or push is performed here.

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
| Runtime | Tree environment; loaded libraries not recorded | Same launch method, exact runtime identity unverified |
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
| `oflm serve`, A→B→A in one process | Six checks PASS on retained production kernels; candidate NOT MEASURED |
| A before B equals A after B | PASS on retained production kernels only |
| B matches a fresh-process B | PASS on retained production kernels only |
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
`accept-run-batch{1,4}.log`, `accept-isolation.log`, `accept-boundary-*/`,
`capacity-phase2-*/`, `capacity-postphase-*/`.

Server result: `utilities/prefill-block/runs/isolation-aba-20261115-223324/result.json`.
The helper sets its own kernel directory, overriding a shell-only selection.

Not measured: candidate server isolation, HTTP TTFT, independent per-layer
residual oracle, Windows, driver-owned NPU memory, freshly rebuilt tiled exports,
context/numerical capability beyond the recorded prompt lengths. Loaded-library
identity for acceptance/confirmation benchmark processes is unverified.
