# K2-Horizon-7B-Uno: making the cycle pay

Status: Phase 2 **done**; Phases 3-7 **proposed** (2026-10-09, for review). Follows `k2-horizon-7b-uno.md`.

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

**Longer outputs (512 tokens, 2026-10-09):** math 4.10 tokens a cycle, 1.52x (12.60 vs 8.28 tok/s); reason 3.88, 1.50x; code 3.82, 1.32x. 41-60% of cycles accept all three drafts, so L = 4 caps acceptance on real output lengths.

**Profile of what is left (one layer at position 0, ms):** dx 2.54; dxl 3.62. With the activation-table rebuild nulled 3.24; norms nulled 3.55; GEMV arithmetic nulled 2.96; everything nulled 2.59.
- The data-movement floor is decode's whole layer.
- The excess is the 8 main cores' arithmetic. Bit-identity makes the 4-token GEMV run decode's float epilogue (hi / lo split, four fp32 MACs) once per token per 32-wide block, so that work scales with L where the weight work does not.
- The same wall blocks L = 8.

## Two modes, chosen by the user (Phases 3-7)

Exact and fast both ship. They pull the pass different ways, so one dxl layout cannot serve both:
- **exact** keeps every row bit-identical to decode, so Uno's output is plain NPU decode's, token for token. That makes it the reference mode.
- **fast** drops decode's per-row arithmetic for a GEMV that multiplies a weight chunk against every row at once. Rows get cheap, which is what L = 8 needs. The cost is an output that can differ from plain decode wherever the top two logits are within rounding of each other.

| mode | rows (L) | output | expected (short / 512-token outputs) |
|---|---|---|---|
| exact | 4 | identical to plain decode | ~1.4x / ~1.7x (after Phase 5) |
| fast | 4 | greedy over its own verify logits; differs from plain decode only at near-ties | ~1.5x / ~1.8x |
| fast | 8 | the same | ~1.6-2x / ~2-2.4x, if L = 8's acceptance holds on long outputs |
| off | | plain decode | 1x |

Every figure above is a projection from today's cost model, not a measurement. Phase 7 replaces them.

### 3. The switch (first: everything after ships behind it)
Same shape as OPEN-PREFILL-MODE (#181), so the two read alike:
- **Kernel set:** `rows` keeps the exact pass; `rows_variants` carries named alternatives (`fast4`, `fast8`), each a complete set of `kernel` / `lora_kernel` / `head` / `l`.
- **Load time:**
  - `oflm run|serve|bench <model> --uno exact|fast|off` and `--uno-block 4|8`. The block size only applies to fast, and asking for exact at 8 is refused at parse.
  - These set `OFLM_OPEN_UNO_MODE` / `OFLM_OPEN_UNO_BLOCK`; `open_qwen36_cli --uno-mode` does the same for A/B runs.
  - The choice is made before the kernel list, so only the chosen variant's xclbins load.
  - A set without the asked-for variant logs that it is falling back to `exact` and runs exact.
- **Per request:** `"uno": false` in a chat request decodes that request plainly. No reload is needed, and it gives a client a reproducible answer whatever mode the server runs.
- **Core:** `uno_cycle` and the CLI loop become L-generic (today they assume `rows_l()` but were only run at 4).
- Cost: ~2 days. If #181 merges first, this reuses its `select_*` shape and flag parsing rather than copying them.

### 4. Prototype the two GEMVs in `gl` before building either pass
`designs/dxl/gl.py` runs the L-row GEMV alone; both candidates get measured there first, at L = 4 and 8.
- **Exact, 2 rows a core:** gemv_q4_tile's arithmetic stays (bit-identity), but each core carries 2 of the 4 rows, on 16 cores fed by one broadcast weight stream per column.
  - The integer mmul costs the same either way; only the per-row float epilogue halves. So this is worth Phase 5 only if the epilogue is most of a chunk's time.
- **Fast:** candidates, chosen by speed and by logits corr against fp64:
  - (a) Dequantize each 32 × 32 block to bf16 once, then bf16 mmul across the rows.
  - (b) The same with the activation as a bf16 hi/lo pair (two mmuls) for accuracy.
  - (c) Keep the int16 × uint8 mmul but scale the activation once per 256-wide k-tile, so the float epilogue runs per k-tile instead of per 32-block. q4_1's per-32 weight scales still need a per-block multiply, so this only helps if that multiply can ride the mmul.
  - Bar: corr ≥ 0.99999 per row against the fp64 replica on a real layer, and the arithmetic at L = 8 under the data-movement floor.
- Cost: 2-3 days. No spec impact.

### 5. Exact mode: 16 GEMV cores (if Phase 4 says it pays)
- **Tiles:**
  - The main cores move to rows 2 and 3, two per column: rows 0-1 and rows 2-3 on one broadcast weight stream.
  - Attention shrinks to groups of 2 cores (16 heads each); 32 tiles do not hold 16 GEMV, 16 attention and the ln core.
  - Each column's two y elements join in a memtile, so the shim output channels stay at 14 of 16.
- **Trade:** short context gets cheaper (the pass ~1.34x → ~1.1x a step). A group of 2 cores pays ~1.5-2x decode's attention arithmetic, so the gain shrinks with context, but the pass is still nowhere near the old 4x.
- **Also here (both modes):**
  - Merge the q / k / v jobs and the gate / up pairs so each slice's activation tables are built once per stage, not once per tile. That is 288 rebuilds a layer down to ~144, about 0.15 ms.
  - Overlap the draft's LoRA A band with the stage before it where the input allows.
- OPEN-DECODE-ROWS keeps its bar; the layout text changes.

### 6. Fast mode: fast4 and fast8
- **GEMV:** Phase 4's fast GEMV, 8 main cores.
- **Attention:** fast8 has 8 row groups of 2 cores (16 cores). fast4 keeps 4 groups of 4.
  - The attention kernels are dx's, unchanged, so attention stays decode's arithmetic in both.
- **Head:** `lmhl` at 8 rows on the fast GEMV.
- **Draft:** the LoRA's A / B bands through the same fast GEMV. `uno.q4nx` is unchanged, since the adapter does not depend on L.
- **New OPEN-DECODE-ROWS-FAST:**
  - Rows within corr ≥ 0.99999 of decode's logits.
  - Argmax equal to decode's except where decode's top two are within 0.05 logits.
- **Modified OPEN-UNO-DECODE:**
  - fast output equals greedy over the fast verify logits; its own exactness check is a fast L-row pass at the same positions.
  - Against plain decode, the first divergence must be a near-tie.
- Cost: 1.5-2 weeks.

### 7. Defaults, by measurement
- The six short prompts, the three 512-token outputs and the 1k / 2k prompts, in every mode, on a quiet machine. The figures replace the projections above.
- Then the default `--uno`. I would expect `fast` with the block size that measures best, and `exact` named as the reproducible option, but this is your call with the numbers in hand.
- The skill and the spec record the table.

## Spec impact (Phases 3-7)
| ID | | Verification |
|---|---|---|
| OPEN-UNO-MODE | new: the switch, the per-request opt-out, the fallback | test: manifest variant selection, flag parsing; manual: each mode loads and decodes |
| OPEN-DECODE-ROWS-FAST | new: fast rows' tolerance | manual: harness layer vs fp64 and dx; whole-model rows check against the tolerance |
| OPEN-DECODE-ROWS | modified: the exact variant's layout (Phase 5) | unchanged bar |
| OPEN-UNO-DECODE | modified: what each mode guarantees | manual: the Phase 7 runs |
| OPEN-UNO-LORA | modified: the draft through the fast GEMV | manual: draft layer vs fp64 |

## Cost and order
3 → 4 → 6 → 5 → 7, about four weeks. Fast before exact's rework: it has the higher ceiling, and its numbers decide whether Phase 5 is worth a week.

**Recommendation on #184:** take it out of draft and land it as it is now. That is K2, serving, and Uno exact at 1.19-1.52x with no regression at any context. Phases 3-7 then go in a PR stacked on it. #184 is already large, and four weeks of kernel work on top would make it unreviewable.

## Not in this plan
- **Retraining or distilling the LoRA.** IFM ships no K2 training recipe.
- **Fusing draft and verify into one forward.** Draft rows appended to a verify pass are only usable when every draft is accepted. Even at 41-60% fully accepted, an extra 4-8 rows a pass loses to the second pass at today's row cost. Revisit after fast8 if rows get cheap enough.
- **Tree verification.** After fast8, if rows are cheap.
- **Reusing unaccepted drafts as the noise inputs.** A CPU-oracle experiment (`utilities/uno-ref`) can be run at any time; it is not on the critical path.
