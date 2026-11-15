# PR #3 acceptance — bounded Q4 GEMM DMA batching

## Recommendation: **KEEP DRAFT** (opt-in, correctness accepted, performance not established)

The candidate is numerically **byte-identical** to the control and passes every
integration gate. Its performance gain is real at 256/512/4096 tokens but small,
and at 1024 tokens it failed its own pre-specified gate. The required
before/after performance claim for PR #3 is therefore **not established**, so
the change stays opt-in with the default unchanged.

Nothing here enables the candidate, changes a numerical tolerance, changes
model precision, or alters the default DMA setting.

## What was compared

| Item | Before | After |
|---|---|---|
| Kernel set | `model-batch1` (`GQP_DMA_BATCH=1`) | `model-batch4` (`GQP_DMA_BATCH=4`) |
| Manifest | identical (`744879192c49…`) | identical |
| Binary | `open_qwen36_cli` / `oflm` same build | same build |
| Runtime | XRT 2.25.00 core + xdna plugin | same |
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
| Per-layer DeltaNet conv + state | 64 layers |
| Per-layer attention KV | 16 layers |
| Logits rows (prompt + 16 decode) | 17, cosine 1.0, max_abs 0.0 |
| Greedy continuation | identical 16 IDs |
| Logits bytes | **byte-exact** |
| State slices | **byte-exact** |

Thresholds unchanged (cosine > 0.9999, residual maxrel < 0.005); the candidate
did not merely meet them, it reproduced the control exactly.

## Prefill performance — mixed

Five alternating pairs per length, one warm-up pair, same binary and method
(`omp_policy.py` kernel mode).

| Tokens | Before median | After median | Change | Pairs faster | Verdict |
|---|---:|---:|---:|---:|---|
| 256 | 29 432 ms | 28 759 ms | **−2.29%** | 5/5 | PASS |
| 512 | 59 365 ms | 58 186 ms | **−1.99%** | 5/5 | PASS |
| 1024 | 118 743 ms | 118 029 ms | **−0.60%** | 4/5 | **FAIL** |
| 3000 | 377 565 ms | 369 280 ms | **−2.05%** | 2/2 pairs | (supporting) |

- 256 and 512 pass the project's paired gate (all five pairs faster; one-sided
  sign test p = 1/32).
- **1024 fails**: only four of five pairs improved, and the −0.60% median
  (711 ms) is inside the observed run-to-run spread (before 118.3–121.1 s,
  after 116.8–119.6 s). This is a genuine gate failure, not an infrastructure
  error: every process exited 0 and produced identical output.
- The 4K point was measured separately (two pairs per arm, alternating) because
  the standard harness only covers the three configured lengths. It shows
  −2.05% with consistent direction, so the gain does **not** decay with prompt
  length; it is the 1024-token measurement that is noise-dominated.
- A secondary diagnostic that ran both 4K arms against the **same** default
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
| ~4K context (3000-token prompt) | 1.654 tok/s | 1.647 tok/s | −0.4% |

All within the pre-existing ±3% decode gate. No systematic decode cost.

## Memory — unchanged

Peak process RSS did not move: 34 278 168 → 34 278 116 KiB at 256 (−0.00%),
34 277 296 → 34 277 208 at 512, 34 277 432 → 34 277 664 at 1024. Driver-owned
NPU memory was **NOT MEASURED** (no supported counter on this stack).

## Integration — PASS

| Gate | Result |
|---|---|
| `oflm run`, both arms | exit 0, 64/64 layers, identical answer `RED` |
| `oflm serve`, A→B→A in one process | PASS on all 6 applicable checks |
| A before B equals A after B | PASS (byte-identical answer) |
| B matches a fresh-process B | PASS |
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

Passive checks only, at the start and end of the campaign:

| Point | Capacity | Kernel errors |
|---|---|---|
| Start of acceptance | 16, 16, 16 | 0 |
| After all phases | 16, 16, 16 | 0 |

Same boot (`9285622f…`). The 16→8 transition observed earlier in this
investigation did not reproduce during acceptance, despite hours of full-depth,
4K-context and server work. It remains an **unresolved intermittent runtime
issue**, not a firmware defect and not a GEMM regression. A separate fresh-boot
diagnostic run also completed without a decrease (`b4ee9f22`).

## Why KEEP DRAFT rather than MERGE

1. The 1024-token prefill point fails the acceptance gate.
2. A gain of ~2% on isolated prefill, with decode and memory flat, is below the
   "reproducible end-to-end performance gain" that justifies changing shipped
   kernel schedules.
3. No target-platform validation beyond this single machine, and no Windows
   measurement.

The candidate is safe — exact numerics, no decode or memory cost, no
integration regressions — so it may be taken forward if a later measurement
establishes a robust gain, but that would need a re-run, not a re-interpretation.

## What would change the recommendation

- Re-run 1024 tokens with more pairs (≥9) or a quieter host to separate a real
  ~0.6% effect from run-to-run spread; a passing 1024 would give three of three
  configured lengths.
- Explain the length dependence, if any is real: the DMA batching groups
  `(row_group, token_tile)` visits, and at a fixed 256-token block the number of
  blocks grows with prompt length, so the saving should scale with blocks.
  Measured absolute savings (1.18 s at 512, 0.71 s at 1024, 8.3 s over 12 blocks
  at 3000) neither scale cleanly nor vanish, which is itself worth understanding
  before promotion.
- Complete Windows and driver-owned memory measurement.

## Evidence

`Models/qwen38-27b/pr3-gemm/`: `accept-full64/`, `accept-e2e-256/`,
`accept-e2e-512/`, `accept-e2e-1024/`, `accept-decode4k-valid.json`,
`accept-run-batch{1,4}.log`, `accept-isolation.log`, `accept-boundary-*/`,
`capacity-phase2-*/`, `capacity-postphase-*/`.

Not measured: Windows, driver-owned NPU memory, freshly rebuilt tiled exports,
context/numerical capability beyond 3000-token prompts.
