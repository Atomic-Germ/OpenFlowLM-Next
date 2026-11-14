# Prefill performance: budget, dispatch table, and what is actually slow

Two tools, two different questions.

| tool | question |
|---|---|
| `measure.py` | where did the wall time go? (stage budget, median of repeats) |
| `dispatch.py` | what was the NPU asked to do, and at what efficiency? |

The second exists because the first one lies by omission. A column reading "host blocked
on GEMM: 24.5 s" sounds like a synchronization problem. Dividing the FLOPs by it says
otherwise.

## Phase 21 result — Qwen3.8-27B, 513 tokens, block route

`Models/qwen38-27b/prefill-perf/dispatch-v1/dispatch-513.json`

```
tokens 513   total 103945.5 ms   coverage 89.8% (min 90%) -> NOT COVERED
  GEMM 74878.6 ms (72.0%)   host 18461.9 ms   unexplained 10605.0 ms
  dispatch column 74906.9 ms, agrees with the GEMM column
  per CALL (one 256-token block)      calls  GFLOP/call  TFLOP/s   %peak@16
  gemm_n14336_k5120             96       37.6    2.0071     12.5%
  gemm_n16384_k5120            288       42.9    1.9886     12.4%
  gemm_n34816_k5120            384       91.3    2.1260     13.3%
  gemm_n5120_k17408            384       45.6    2.1050     13.2%
  gemm_n5120_k6144             384       16.1    1.7701     11.1%
  paired `_low` products: 38557.0 of 74878.6 ms (51.5%)
  prompt throughput: 3 blocks for 513 tokens (1.50x padding),
                     achieved 1.996 TFLOP/s, useful 0.667 TFLOP/s
```

Four findings, in the order they change what to fix:

1. **The GEMMs sustain ~2.0 TFLOP/s per dispatch, ~12.5 % of a 16-TOPS part.** That is
   uniform across all five shapes (1.77–2.13), so it is not one bad tiling. It is the
   arithmetic-to-DMA ratio of the q4_1 route: one 64×64 weight band is dequantised
   on-core and then used against a 32-wide activation tile, so every weight byte is
   touched ~1.2 times instead of ~256 times at a prefill-friendly tile.

2. **The paired correction is 51.5 % of the GEMM column.** The four-product route buys
   precision and costs half the GEMM time. This is the plan's first suspect, confirmed
   by measurement rather than assumed.

3. **Block padding costs 1.50×.** 513 tokens occupy three 256-wide blocks = 768 slots,
   so a third of the GEMM work is padding. Useful throughput is 0.667 TFLOP/s, a third
   of the achieved 1.996.

4. **Coverage is 89.8 %, short of the plan's 90 % by 0.2 points.** The gap is
   `residue` 10576.6 ms: host sub-stages the engine does not clock individually (the
   named fine stages sum to more than the host column, so they overlap). It is a
   measurement-completeness gap, not an unexplained performance mystery, and it is
   reported as NOT COVERED rather than rounded up.

## What that means for Phase 22

Ranked by measured size, not by plausibility:

| lever | outcome | evidence |
|---|---|---|
| drop the paired `_low` products entirely | −49 % wall, **rejected: 301/513 below the gate** | `phase22-levers-v2.json` |
| drop only `W_lo X_lo` (`three_term`) | −23 % wall, **1/513 below the gate** | same |
| layer-major schedule | **no lever** — identical call counts and time | same |
| wider compiled block | **no lever** — 513 tokens already costs 3 calls | same |
| larger activation tile per weight band | **blocked** — does not fit L1 | same |
| dequant / gather | 14.3 % / ~2 %, dequant already 2.8× faster | same |

## Lever 1 measured, and rejected: `--prefill-precision weight_high`

The first lever was implemented as a real setting (`prefill_precision`: `paired` /
`weight_high` / `single`, default `paired`) and measured on hardware at 513 tokens over
all 64 layers, one binary, the only difference being the mode:

| | GEMM ms | wall ms | min corr vs reference | failing the 0.9999 gate |
|---|---|---|---|---|
| `paired` | 74575.7 | 102721.2 | 0.999911 | **0 / 513** |
| `weight_high` | 34201.3 | 52134.6 | 0.490848 | **301 / 513** |

The saving is real and is exactly the `_low` share: GEMM −54.1 % against a measured
`_low` share of 51.5 %, and 1.97× on wall time. **The accuracy cost is not acceptable.**
`weight_high` is left in the engine as an explicit opt-in (it is the only way to measure
this, and it is what a future partial-residual scheme can be compared against) but it is
not a candidate for the default at this prompt length.

**Why the per-projection argument was wrong.** `_low` carries only the BF16 rounding of
the already-dequantised weight, so against the int4 lattice error of a single projection
it looks negligible — and that is how it was argued before it was measured. Prefill sums
64 layers of projections, and the dropped term is a *systematic* rounding whose sign is
correlated with the weight, so it accumulates coherently instead of averaging out. The
p50 defect rises from 1.1e-06 to 2.3e-04 — three orders of magnitude — while the
per-projection estimate said it should not move. A per-operation error budget does not
compose additively, and this is the measurement that shows it. The paired route's 2×
GEMM cost is what buys the correlation.

## Phase 23: the `three_term` failure is role-localised, and STILL rejected

One bounded experiment, as the plan specifies. `three_term` for one projection ROLE at a
time, `paired` everywhere else, 513 reference positions, no position specialisation.
Tool: [`three_term_roles.py`](three_term_roles.py); data:
[`phase23-attribution.json`](phase23-attribution.json).

| variant | stage ms | GEMM ms | min cos | failures | worst | e2e gain |
|---|---|---|---|---|---|---|
| control `paired` | 116482.8 | 74711.0 | 0.999911 | 0 | P494 | — |
| `qkvz` | 111454.9 | 71332.8 | 0.999976 | 0 | P399 | +4.3 % |
| `attn_in` | 114319.2 | 73563.5 | 0.999951 | 0 | P459 | +1.9 % |
| `out` | 113303.8 | 72963.9 | 0.999920 | 0 | P494 | +2.7 % |
| `ffn_up` | 106587.4 | 66535.6 | 0.999915 | 0 | P494 | +8.5 % |
| **`ffn_down`** | 110865.0 | 70803.1 | 0.999868 | **1** | P399 | +4.8 % |
| all but `ffn_down` | 98430.0 | 60747.9 | 0.999888 | **1** | P494 | **+15.5 %** |

**The failure is structural, not token-specific.** Four of five roles pass with zero
failures, and the failure sits only in `ffn_down`. The worst position also moves between
variants (P399, P459, P494), so nothing here is one unlucky position.

**And it does not compose.** Excluding the guilty role — the cleanest policy the export
can express — is 15.5 % faster and still fails once. Errors from the four clean roles
accumulate: `qkvz` alone clears the gate at 0.999976 with 24× the margin, yet together
they push a different position under it. A per-role budget does not compose any more
than a per-projection one did, which is the same lesson Phase 22 measured.

**Decision: CLOSE.** The promotion gate needs zero failures AND >= 10 % end-to-end
gain. The >= 10 % half is reachable (+15.5 %); the zero-failure half is not, in any role
subset the export allows. `paired` is frozen permanently as the production prefill
arithmetic, per the plan's own rule for a failed bounded cycle.

**Why the other six rows of the plan's role list are absent.** They are not separable:

| asked for | why untestable |
|---|---|
| `qkv` / `z` separately | one fused `n16384_k5120` GEMM |
| `q` / `k` / `v` / `gate` separately | one fused `n14336_k5120` GEMM |
| `up` / `gate` separately | one fused `n34816_k5120` GEMM |
| DeltaNet alpha/beta | runs in the glue kernel, not a projection GEMM |
| LM head | outside block prefill; tail-only, no paired/three_term variant |

Splitting any of them would mean repacking the model, which is a separate project and
not what this bounded experiment authorises.

## Lever 2 measured earlier: `--prefill-precision three_term` (whole run)

`three_term` keeps `W_hi X_hi + W_hi X_lo + W_lo X_hi` and drops only `W_lo X_lo`. With
both operands split, `|W_lo| ~ 2^-9|W|` and `|X_lo| ~ 2^-9|X|`, so that term is ~2^-18 of
the projection — 512× smaller than either term `weight_high` gives up.

| | GEMM ms | wall ms | min corr | failing the gate |
|---|---|---|---|---|
| `paired` | 74575.7 | 102721.2 | 0.999911 | 0 / 513 |
| `three_term` | 56470.6 | 78987.6 | 0.999891 | **1 / 513** (P494) |
| `weight_high` | 34201.3 | 52134.6 | 0.490848 | 301 / 513 |

−23.1 % wall, and the one failure is P494 — the same position that is already the worst
in `paired` (0.999911, a 1.1e-5 margin). The gate is **not** relaxed to take this: it is
an explicit opt-in and stays one. Worth knowing: the margin on that position is thinner
than the term `paired` spends 24 % of its wall time buying.

## Every other lever measured and closed

| lever | result |
|---|---|
| layer-major | identical: 96 high + 96 `_low` per shape, 103566.7 ms against 102721.2 |
| wider block | no gain — 513 tokens already costs 3 calls; the 1.50× is a short-prompt tail, which is what `OFLM_OPEN_GEMM_BLOCK_MIN` exists for |
| `GQP_TILE_N=64` | refused by the toolchain: `aie.tile op allocated buffers exceeded available memory` |
| sub-core-tile K slicing | the A refetch cost sums to `NRB × Nslices = N/(64·8)` — unchanged, so it cannot pay |
| `b_reuse` | the design documents it as failing at 8 columns (`repeat_count` on a shim tile, and the mem-tile BD pool) |
| submit-ahead | already on (`submit_ahead_ = 1` by default) |

**Why the GEMM cannot go much faster here.** A single-kernel ablation
(`GQP_NULL_DEQUANT` / `GQP_NULL_GATHER`, 5 runs each, `n14336_k5120_t256`) measures the
matmul at **85.7 %** of a call: base 16.94 ms, dequant nulled 14.52, gather nulled 16.24.
A core's operand tile is `m=64, n=32` per k-step — 4096 B of A and 4096 B of B — which is
`n/m = 1/2` MACs per byte, and the tile **cannot grow**: `tile_n=64` exceeds L1. So the
arithmetic intensity is fixed by the 64 KB local memory, and 2.0 TFLOP/s (~12.5 % of
peak) is this design's ceiling rather than a scheduling mistake. Beating it means a
different datapath (weight-stationary, or int8/fp8 weights), not a tuning pass.

## Reproducing

```bash
XILINX_XRT=/opt/xilinx/xrt LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
OPENBLAS_NUM_THREADS=1 OFLM_OPEN_DISPATCH_LOG=1 \
  Models/qwen38-27b/post-prefill-production/plain-v2/build/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-v2/kernels-model \
  --ids-file <256 ids> --max-ctx 2048 --max-tokens 1 --gemm-block \
  2> dispatch.log

ironvenv/bin/python utilities/prefill-perf/dispatch.py --log dispatch.log \
  --out dispatch-513.json
```

`--gemm-block` is what makes the engine print the dispatch table: it is emitted from
the block-route branch, and a run without it takes the same path but prints no table.
That is a property of the CLI's logging, not of the route.

## The bug this tool was written with, and why it is pinned

The first version divided the PROMPT's FLOPs by the sum of every call's time, and
reported **0.011 TFLOP/s** for a GEMM that sustains **2.0**. A 513-token prompt makes
three 256-wide calls per GEMM, so the aggregate time is ~3× one call's and the FLOPs
were for the whole prompt. The correction is per CALL against the compiled block width;
`test_each_call_carries_one_compiled_block_not_the_whole_prompt` pins it. Had that
number been believed, it would have argued that the GEMM was hopeless rather than
tile-bound -- the opposite of the fixable conclusion.
