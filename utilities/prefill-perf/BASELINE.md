# Frozen correctness baseline: paired Q4 prefill

**Status: FROZEN 2026-11-01.** This document closes the prefill tuning phase. What it
pins is the *numerics*, not the speed: the paired four-product Q4 prefill route is the
correctness baseline that every future datapath is judged against. Nothing here is a
claim that the current speed is good.

## What is frozen

| item | value | where |
|---|---|---|
| default `prefill_precision` | `paired` | [`settings.json`](../../src/config/settings.json) |
| the arithmetic | `(W_hi X_hi + W_hi X_lo) + (W_lo X_hi + W_lo X_lo)` | [`prefill_precision.hpp`](../../src/open_qwen36/prefill_precision.hpp) |
| logit gate | Pearson ≥ **0.9999** over `real_vocab = 248077` | [`utilities/prefill-long-context/check.py`](../prefill-long-context/check.py) |
| reference | FP64-derived per-position logits, 513 positions | `Models/qwen38-27b/prefill-long/boundaries-513/513/` |

The gate is **not** relaxed for any speedup. `three_term` and `weight_high` exist as
explicit opt-ins purely so their cost can be measured; neither may become the default,
and `test_the_default_is_the_four_product_sum` fails the suite if one tries.

## Baseline measurements

Production route, 513-token prompt, all 64 layers, `--gemm-block`, `T=256` compiled.

```
total      102721.2 ms
  GEMM      74575.7   72.6 %
  host      17507.9   17.0 %
  other     10637.6   10.4 %
```

GEMM per shape (96 high + 96 `_low` calls each):

| shape | calls | ms | mean ms | TFLOP/s |
|---|---|---|---|---|
| `gemm_n14336_k5120` | 96+96 | 1796.2 + 1933.3 | 18.71 / 20.14 | 2.01 / 1.88 |
| `gemm_n16384_k5120` | 288+288 | 6204.8 + 6471.7 | 21.54 / 22.47 | 1.99 / 1.90 |
| `gemm_n34816_k5120` | 384+384 | 16400.9 + 17496.9 | 42.71 / 45.57 | 2.13 / 1.99 |
| `gemm_n5120_k17408` | 384+384 | 8278.1 + 8848.9 | 21.56 / 23.04 | 2.11 / 1.97 |
| `gemm_n5120_k6144` | 384+384 | 3452.9 + 3692.0 | 8.99 / 9.61 | 1.77 / 1.66 |

`_low` share of the GEMM column: **51.5 %** (38442.8 of 74575.7 ms).

Accuracy of the frozen route: **0 of 513** positions below the gate, min correlation
0.999911 (P494), p50 defect 1.129e-06.

Reproduce:

```bash
source utilities/env.sh
CLI=Models/qwen38-27b/prefill-perf/switch-v1/build/open_qwen36_cli
OFLM_OPEN_DISPATCH_LOG=1 OFLM_OPEN_PREFILL_PRECISION=paired \
  $CLI --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
       --kernels Models/qwen38-27b/production-v2/kernels-model \
       --ids-file Models/qwen38-27b/prefill-perf/dispatch-v1/ids256.txt \
       --max-ctx 2048 --max-tokens 1 --gemm-block 2> paired.log
ironvenv/bin/python utilities/prefill-perf/dispatch.py --log paired.log --tokens 513
```

## Single-kernel characterisation (the number the next section builds on)

`gemm_q4_prefill`, `n14336_k5120_t256`, 5 runs per configuration, timing-only ablations
(`GQP_NULL_DEQUANT` / `GQP_NULL_GATHER` make the output garbage; they are never combined
with a correctness run):

| configuration | ms | share of base |
|---|---|---|
| base | 16.94 | 100 % |
| gather nulled (`GQP_NULL_GATHER`) | 16.24 | 95.9 % |
| **dequant nulled (`GQP_NULL_DEQUANT`)** | **14.52** | **85.7 %** |
| both nulled | — | ~85.7 % is the matmul and the streams |

So a call is 85.7 % "the matmul and the streams". That label is deliberately vague,
because the arithmetic says the MACs alone cannot account for it: one core does
4 × (64×64×32) MAC per band group, and 56 band groups in 302 µs — but 2048 MAC-cycles
per band group is ~2 µs at 1 GHz. The core is spending ~1.35 µs per
`(ky, band)` acquire/matmul pair, which is **small-DMA latency**, not MAC throughput.

That is the hypothesis the weight-stationary POC tests, and it is the reason the next
step is a reorder of the loop rather than a tuning of the tiles: the same 224
acquire/matmul pairs per core per dispatch stay, but with one dequantized weight tile
reused across R activation microtiles they can be issued back-to-back.

## Why the phase closes here

Every lever that does not change the datapath was measured and closed
([`README.md`](README.md)): layer-major is bit-identical in cost, a wider block gains
nothing (513 tokens already costs 3 calls), `GQP_TILE_N=64` does not fit L1, sub-core-tile
K slicing conserves the A refetch exactly, `b_reuse` is documented as failing at 8
columns, and submit-ahead is already on. The remaining levers change the arithmetic, and
both are quantified and gated.

The next step is therefore a **datapath** change, taken as a standalone proof of concept
with its own gate, not as a tuning pass inside the runtime.

## Weight-stationary POC: measured, and STOPPED

[`ws_poc.py`](../../open_kernels/designs/gemm_q4_prefill/ws_poc.py) reorders the same kernel
(no tile change, `tile_n` stays 32) so one dequantized weight tile feeds R activation
microtiles, R = `T / (tile_n * n_aie_cols)`. Measured with
[`ws_measure.py`](ws_measure.py), which gives each variant its own dump file — `compare.py`
reads one shared `y_<tag>.bin`, so it scores the last run twice.

| | array ms (median) | cosine vs fp64 | verdict |
|---|---|---|---|
| control, `gemm_q4_prefill` at T=2048 | **136.206** | 0.999999 | PASS |
| POC R=1 | 9.4 / 7.1 / **2043** (state 8 hang) | 0.009117 | **FAIL** |

The POC's fast dispatches are *partial*, not fast: cosine 0.009 because the output is
incomplete. **Why it is wrong and not merely slow.** One output microtile `[64,32]`
accumulates over the whole K. The baseline keeps one accumulator in L1 across all 20
band groups of a T-tile — its `core_fn` acquires C once per T-tile, *outside* the band
loop. A band-stationary reorder releases each weight band before the next arrives, so
with one resident accumulator it cannot sum 20 bands into one output. The only correct
form holds **R accumulators**, and that is what the L1 budget forbids:

| resident form | scratch | accumulators | fifos | total | fits 64 KB |
|---|---|---|---|---|---|
| baseline (1 accum, depth 2) | 8 K | 8 K | 45 K | **65536** | exactly, 0 B headroom |
| output-stationary R=1 | 8 K | 8 K | 22.5 K | 43008 | yes |
| output-stationary R=2 | 8 K | 16 K | 22.5 K | 51200 | yes |
| output-stationary R=4 | 8 K | 32 K | 22.5 K | 67584 | **no** |
| output-stationary R=8 | 8 K | 64 K | 22.5 K | 100352 | **no** |

Build failures confirmed the arithmetic: four scratch tiles with four accumulators fail
`allocated buffers exceeded available memory`; R=8 additionally fails
`Too many simultaneously active buffer descriptors on tile (3,0), which supports up to 16`.
Forcing the fifo depths to 1 makes it build, which spends exactly the pipelining the
baseline's 136 ms depends on.

**Decision: STOP.** R=2 is the largest R that fits, and the ceiling R=2 can reach is set
by the dequant it removes: the ablation above puts dequant at 14.3 % of a call, so
removing half of it is at most 7.2 % — **1.08×**, below the 1.2× stop threshold. That
bound is arithmetic, not a measurement artefact, so no further build clears it.

**The measured bottleneck, stated plainly.** The GEMM is compute-bound at 2.0 TFLOP/s
(12.5 % of peak) because a `64×32` core tile is `n/m = 1/2` MACs per byte and cannot grow
(`tile_n=64` exceeds L1). Weight re-fetch and dequant are real but small. Anything past
this needs a different datapath — weights resident in a larger local memory, or int8/fp8
weights — not a loop reorder of this one. Evidence:
`Models/qwen38-27b/prefill-perf/switch-v1/ws-poc-v1.json`.

The model runtime was not touched: no code under `src/` changed for this experiment.
