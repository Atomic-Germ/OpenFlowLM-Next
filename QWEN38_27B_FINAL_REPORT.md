# Qwen3.8-27B on open XDNA2 — final report

**Verdict: PASS.** Text-only Qwen3.8-27B runs on the open XDNA2 kernel set, on the shipped
`oflm` binary, with block prefill as the production route.

Machine-readable companion: [`qwen38-production-baseline.json`](utilities/prefill-block/qwen38-production-baseline.json).
Long-context contract: [`long-context-v1.json`](utilities/prefill-block/long-context-v1.json).
Block contract and the Phase 27 closure record: [`BLOCK_CONTRACT.md`](utilities/prefill-block/BLOCK_CONTRACT.md),
[`PHASE27_REPORT.md`](utilities/prefill-block/PHASE27_REPORT.md).

---

## 1. What was proven

| area | result |
|---|---|
| binary | shipped `oflm`, `7bb6c8aa`, sha256 `1315d111…f090`, 31 886 824 B, built by the repo's own Docker path |
| weights | `model.q4nx`, **947 tensors**, **19 844 633 616 B = 19.845 GB**, no embedded metadata |
| prefill route | block loop by default at 96; layer-major only for routed-expert sets or a manifest that declares `block_loop_preferred` |
| prefill correctness | 64-layer prefill → 16-token decode oracle, **16/16**, min cosine **0.999999812**, max_abs **6.914e-03**, **no new tolerance** |
| decode | **2.087 tok/s** median (range 2.05–2.09), flat in prompt length, load and prefill excluded |
| prefill cost | 256 / 512 / 1024 → TTFT **39.5 / 73.2 / 143.0 s**, **6.68 / 7.10 / 7.22** prompt tok/s |
| context, capacity | **262144** — buffers allocate, the context object accepts the length |
| context, addressing | **262144** — position arithmetic correct to position **262143** |
| context, end to end | CLI **4095**, serve **4093** engine tokens |
| generation limit | prompt ≤ `ctx_len − 5` (not thinking) / `ctx_len − 3` (thinking); refused explicitly, never a throw |
| regression | **2350 passed, 172 skipped, 0 failed**; settings CI OK against merged `origin/main` |

## 2. What is NOT claimed

- **Vision is not supported.** Qwen3.8-27B here is a text model; no image path was built or tested.
- **256K is not end-to-end validated.** 262144 is a capacity and addressing result. The largest
  prompt that actually went through the model is **4095** engine tokens.
- **Long-context attention is not optimized.** It was *selected* as the next target, not fixed.
  At 4K it is 12.3% of prefill time, fitting `1.628 ms/position` with `R² = 0.99768`; the 20%
  crossing is ≈7.6K.
- **No arithmetic or datapath change.** INT8 / INT4 / FP8, `three_term`, `weight_high` and
  weight-stationary were all evaluated and none shipped.
- **No speedup in this phase.** Phase 31 changed correctness accounting only.

## 3. The one real bug this phase found

`Qwen3_5VL::generate` does not begin with a sampled token. It steps a **think preamble** first:

| branch | rows | ids |
|---|---|---|
| thinking | **2** | `think_start_id`, `198` |
| not thinking | **4** | `think_start_id`, `271`, `think_end_id`, `271` |

Two things were wrong because of it, and both were invisible for the same reason — the counter
and the engine position had silently diverged:

1. The loop's budget guard ran **after** the preamble had already written positions, so a
   near-limit prompt threw `position N reached the context capacity N`, which the server
   surfaced as a generic `Internal error`.
2. `total_tokens` was **never advanced** by the preamble, so the guard was measuring a stale
   position. This is why guessing "leave N slots" failed at **N = 2, 3, 4 and 5** — each guess
   just moved the observed boundary.

The fix is structural, not a constant: `Qwen3_5VL::generation_preamble_rows()` is the single
source for the guard, for the counter advance, and for the template trim, and the guard sits
before the first preamble step.

Measured at a 512 context, both branches, both sides of the boundary
(`runs/think-boundary-512-20261103-015405/`, 8/8):

| branch | prompt | need | observed |
|---|---|---|---|
| not thinking | **507** | 5 | **generates**, ends at `total == MAX_L` |
| not thinking | **508** | 5 | **refused before stepping** — names 507 as the largest usable prompt |
| thinking | **509** | 3 | **generates**, ends at `total == MAX_L` |
| thinking | **510** | 3 | **refused before stepping** — names 509 |

The refused rows carry **no `ctx-trace step` line at all**, so no position was written. Over
HTTP the refusal is `400 {"message":"Max length reached!","type":"model_error"}`.

## 4. A correction to Phase 29's explanation

Phase 29 blamed the serve gap on `_shared_insert` sampling the first token as part of prefill.
That was **partly right and did not account for the size of the gap**: fixing it moves the
boundary by one slot, while the observed serve limit was several slots below the CLI's. The
preamble is the rest of it. Phase 29's reverted `+1` was the right instinct on the wrong layer;
the shipped fix is in `generate()`, where the rows are actually stepped.

## 5. Route policy (Phase 28, still the production rule)

`layer_major_ok()` is purely mechanical — true iff every layer's `gemm_block.kind` is `linear`
or `full` — so it says nothing about which route is *better*. The measured difference is one
thing: **layer-major pads a tail to `TOT = B*T` whole blocks; the block loop steps it.**

| engine tokens | layer-major vs block loop |
|---|---|
| 264 | **1.80×** faster |
| 2056 | 1.11× |
| 256 (exact multiple) | **1.000** — parity |
| 512 (exact multiple) | 1.024 |

So the dense default is the **block loop at 96**, dense sets only take layer-major when the
manifest declares `block_loop_preferred`, and routed-expert sets keep it as their owner.

## 6. Context: four numbers, never interchanged

```
capacity  262144   buffers allocate, the context object accepts the length
addressing 262144  position arithmetic correct to 262143
CLI e2e      4095  a real prompt through the real model, then decoded
serve e2e    4093  the same, through the API
```

KV is `8192 B` per token per **full** layer (`4 bf16 planes: [K hi, K lo, V hi, V lo]`), so
`8192 × 16 = 131072 B/token`, and **32.00 GiB at 256K**. It is a **device BO**, so host RSS
cannot be used to size it — an earlier probe that tried reported a −274 B/token slope and was
correctly labelled inconclusive rather than passed.

`max_prefill_len` stays **4096** and `default_context_length` stays **32768**: 8K and 16K did
not pass the end-to-end gate, so there is no evidence to raise either, and a 256K default
would allocate 32 GiB of KV for every request.

## 7. Gates

| # | gate | result |
|---|---|---|
| 1 | binary provenance | PASS — `7bb6c8aa`, `1315d111…`, 31 886 824 B |
| 2 | route matrix | PASS 9/9 at 95/96/97, 255/256/257, 511/512/513 |
| 3 | prefill → 16-token oracle | PASS 16/16, cos 0.999999812 |
| 4 | normal `oflm run` | PASS — short sequential and multi-block |
| 5 | normal `oflm serve` | PASS — and over-limit is HTTP 400 `model_error` |
| 6 | request isolation | PASS A → B → A |
| 31.A | non-thinking boundary | PASS 507 / 508 |
| 31.B | thinking boundary | PASS 509 / 510 |
| 31.E | decode rate | PASS 2.087 tok/s |
| 31.F | prefill baseline | PASS 256/512/1024 |
| 31.G | model provenance | PASS 947 tensors, 19.845 GB |
| 31.H | context levels | PASS, kept separate |
| 31.I | support statement | PASS, text-only |
| 31.J | full regression | PASS 2350 / 172 / 0, settings CI OK |

## 8. Reproducing

```bash
source utilities/env.sh                       # XRT, MLIR-AIE + Peano, the tree's paths
python utilities/prefill-block/think_boundary.py --ctx 512
python utilities/prefill-block/route_matrix.py --mode default
python utilities/prefill-block/decode_rate.py --tokens 32
python utilities/prefill-block/oflm_sanity.py
python -m pytest specs/ -q
utilities/check-settings-ci.sh
```

## 9. Where the next work goes, if any

Phase 30 (optional arithmetic / datapath research) is **DEFERRED**, and no optimization was
started here. If long-context matters more than single-request latency, the selected target is
attention: it is the only term that grows quadratically and it crosses 20% of prefill at ≈7.6K.
That is a separate, gated phase with its own correctness oracle — not a continuation of this one.
