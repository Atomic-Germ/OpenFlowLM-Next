# K2-Horizon-7B-Uno: making the cycle pay

Status: **approved, in progress** (2026-10-09; no stopgap: #184 stays draft until this lands). Follows `k2-horizon-7b-uno.md`.

## Where it stands (quiet machine, shared lock's `timing` gate)

| context | plain decode | Uno | Uno cycle | cycle in decode steps | tokens a cycle | Uno vs decode |
|---|---|---|---|---|---|---|
| ~20-150 (6 prompts x 128) | 107 ms/token | 86-109 | 335 ms | 3.13 | 3.47 | **1.11x** |
| ~1000 (64 tokens) | 131 | 193 | 650 | 4.95 | 3.37 | **0.68x** |
| ~2000 (64 tokens) | 157 | 225 | 720 | 4.60 | 3.20 | **0.70x** |

- An L = 4 verify pass costs **1.39x** a decode step at short context (`--rows-check`: 149.6 against 107.3 ms). The 1.17-1.28x in the earlier plan came from a contended machine.
- The draft pass is the rest of the cycle, ~185 ms (**~1.7x**). So there is no missing overhead: both passes are simply expensive.
- **Context is the main problem.** Decode gains ~25 ms per 1000 tokens of context. A pass gains ~4x that, because dxl runs attention one query row at a time (`dxl.py:362`), reading each layer's KV window once per row: 8 window reads a cycle against decode's 1.
  - The model `cycle = 315 + 200·C` ms (C = context in thousands of tokens) fits the short and 2k runs; the 1k run sits ~135 ms above it, unexplained.
  - Break-even with decode is ~0.4-0.5k tokens of context.
- **Consequence for PR #184 as written:** K2 is a reasoning model whose replies run to thousands of tokens, and `oflm` sends every greedy request through Uno. Most real requests get ~30% slower.

## Acceptance: fixed per draft, not per cycle

The LoRA's per-position draft quality is fixed (no retraining here), and we already match IFM's reference (3.47 against the CPU oracle's 3.45 at L = 4). But tokens per cycle is not fixed:
- **Block size.** IFM evaluates at `--diffusion-block-size 8` and reports 2.71 tokens per forward. Their TPF counts both passes (`two_pass_decoding.py`: `forwards += 2`), so that is ~5.4 tokens a cycle on their benchmarks. Our L = 4 caps a cycle at 5.
  - Our CPU oracle gave 3.94 at L = 8 on the six 128-token chat openings. Long reasoning outputs, which are what K2 produces, probably accept more; that is not measured yet.
- **Tree verification.** Their tree sampler (top-k 32, 60 verify nodes) raises accepted length per cycle.

Both cost rows. On a GPU a pass of 8-60 rows costs about one row; here the L-row GEMV is compute-bound past 4 rows (L = 8 runs two tile4 passes, ~1.9x the GEMV). So rows, not acceptance, are the binding constraint, and that is a kernel problem.

## Phases

### 1. Profile a pass by stage
Split the 149.6 ms verify and ~185 ms draft passes into GEMV, per-row ln / attention / silu, head (`lmhl` measured 2x `lm_head_q4` in the harness), and the LoRA's A→z→B dependency, at 150 and 2k context.
- Method: the harness with stage-nulled builds (as `GL_NULL` / `lmhl_null` did), plus `rows_ms` per pass in `--uno`.
- No spec impact. Cost: a day.
- Decides the order of 2 and 3, and the shape of 2. dx's fast attention (4 cores × 8 heads, RB 4, OPEN-ATTN-CONTEXT) streams each position's whole K and V rows (4 KB) once through one shim channel, broadcast to all four cores. Decode's slope (25 ms per 1k positions, 36 layers) is ~5.9 GB/s. Bound by that one channel, reading the window once for all L rows is the fix. Bound by the cores' per-position arithmetic, the L rows need their own cores: 16 of the 32 tiles are idle, and a broadcast window feeds any number of them. Measure: dxl and dx in the harness at pos0 0 / 1024 / 2048, then a dxl build with the attention arithmetic nulled (fifo traffic kept).

**Measured 2026-10-09 (harness, one layer, ms):**

| position | 0 | 1024 | 2048 | 3072 |
|---|---|---|---|---|
| dx (decode) | 2.56 | 3.35 | 4.05 | 4.78 |
| dxl verify (L = 4) | 3.75 | 6.60 | 9.55 | 12.5 |
| dxl, `ATTN_NULL` | 3.75 | | | 7.4-8.2 |

The pass grows 2.9 ms per 1k positions against decode's 0.74 (3.95x). Per row and 1k positions, 0.40 ms is attention arithmetic and 0.32 ms the window walk's streaming and handshakes. So one shared window on the same 4 cores would still pay 4x the arithmetic (~1.9 ms per 1k); the rows need their own cores.

### 2. Row-parallel attention in dxl
- **Layout:** four row groups of 4 attention cores (16; 13 tiles are idle today). Group j is decode's attention for row j: the same kernels, the same h0 split, unchanged.
- **One broadcast `ain` stream:** the four rows' prologues (meta + ptab, q, k, v) in order, then the window [0, pos0 + L) once. Group j skips the other rows' prologue elements, walks the first pos0 + j positions with the pb counts meta computes for position pos0 + j (decode's blocks and singles exactly), skips the remaining L − j positions, then adds its own row through `attn_step_new` as decode does.
  - **Bit-identical to decode by construction**, so OPEN-DECODE-ROWS stays as written.
- **Order:** every group's new K/V row reaches the cache (one `aout` per group, joined in a memtile) before the window fill that reads it is issued.
- **Outputs:** each group's four og elements join in a memtile into one 8 KB row: one drain per row, 4 og channels as today. Shim output channels stay at 14 of 16.
- **Expected:** pass attention slope 2.9 → ~0.75 ms per 1k positions; at 2k, a verify layer ~9.5 → ~5.2 ms.
- **Then Phase 3:** the 3.75 vs 2.56 ms at position 0 (+46%) remains.

**Done 2026-10-09.** Bit-identical first build (layer 0 at pos0 0, 1, 5, 8, 1022; whole-model `--rows-check` PASS). One layer: 3.63 / 4.38 / 5.08 / 5.95 ms at 0 / 1k / 2k / 3k (decode 2.63 / 3.34 / 4.07 / 4.85).

| | before | after |
|---|---|---|
| six prompts × 128 (short context) | 1.11x | **1.19x** (1.05-1.34) |
| ~1k context | 0.68x | **1.21x** (9.27 vs 7.66 tok/s) |
| ~2k context | 0.70x | **1.21x** (7.79 vs 6.46 tok/s) |

All identical to plain decode. Verify pass at short context: 145.9 ms, 1.34x a step.

### 3. Base pass cost
Target verify ≤ 1.15x and draft ≤ 1.3x a step at short context: cycle ~245 ms, ~1.4-1.5x. Candidates, in the order Phase 1 will likely rank them:
- The head: `lmhl` at 2x `lm_head_q4`. 10 KB elements (`lmhl2`) did not help, so the cost is elsewhere in it.
- The serial per-row stages (ln, attention, silu), each one row at a time between GEMVs.
- The draft's LoRA: z = A·x must reach DDR before B's k-tile. That is a dependency bubble per projection group, ×4 groups × 36 layers.
- Per-layer `start_run` / `wait_run` keeps only one layer queued ahead (`core.cpp:3401`); queueing all 36 is free to try.

### 4. Acceptance on real outputs, then rows
- With `utilities/uno-ref`, measure tokens a cycle on long reasoning generations (512-2048 tokens; math, code, chat) at L = 4 and L = 8.
- Also try reusing the previous cycle's unaccepted drafts as the noise inputs. They were drafted in parallel from noise, so they are as good a guess as noise, possibly better.
- If L = 8 gains ≥ 25% tokens a cycle on that set: an 8-row GEMV that dequantizes each chunk once for all 8 rows (today L = 8 re-runs the whole tile4 pass) is the next kernel.
- Tree verification only after rows are cheap.

## Expected outcome

| after | short context | 2k context |
|---|---|---|
| today | 1.11x | 0.70x |
| 2 | ~1.15x | ~1.2x |
| 2 + 3 | ~1.45x | ~1.45x |
| 2 + 3 + 4 (if L = 8 pays) | ~1.6-1.9x | ~1.6-1.9x |

Projections are arithmetic on the cost model above, not measurements.

## Not in this plan
- **Retraining or distilling the LoRA.** IFM ships no K2 training recipe.
- **Fusing draft and verify into one forward.** Draft rows appended to a verify pass are only usable when every draft is accepted (~30% of cycles at L = 4). With rows past 4 costing ~linearly, the arithmetic loses: an 8-row pass at ~2.2 steps gives ~3.4 steps a cycle in expectation, against 3.1 today.
