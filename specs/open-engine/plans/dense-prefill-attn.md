# Dense block prefill: the attention products, and the route's host stages

**Status (2026-10-04):** implemented on `perf/dense-prefill`, `spec.md` updated. The full-model
check passes on Qwen3-8B. **Open:** `oflm-test --llm` through `oflm serve`. Move this plan to
`archive/` once that passes.

## Why

The dense block route ran attention as one `dxB` dispatch per token per layer. On Qwen3-8B
that is 63 % of a 981-token prefill (20.9 of 33.3 s), against 3.1 s for the closed engine. The
attention products (`OPEN-PREFILL-ATTN`) already run the 35B's and Qwen3.5's full-attention
layers. PR #139 made `recipes/dense.py` emit them for dense families, but the engine never
read them there.

## Spec impact

- **OPEN-PREFILL-ATTN (modified).**
  - The dense route runs the products for a layer type whose `attn_block` declares `prep`
    (the one value: `qknorm_rope`). A dense `attn_block` without `prep` is not read; an
    unknown `prep` is refused.
  - A window within `l_max` runs every kv head's scores before any head's values. Bit-identical
    output, two context switches a layer instead of two a head.
  - The 1/sqrt(hd) folding is exact only at head dims 64 and 256.
  - New unit criteria: the dense emission and the dense parser. The procedure gains the dense
    full-model check (`attn_route_check.py`, corr >= 0.9998, the distance the shipped routes
    keep from each other).
  - Result 2026-10-04.
- **OPEN-PREFILL-BATCH (modified):** the `dense` kind's attention is the products where `prep`
  is declared.
- No new requirement IDs. The host-stage change (token rows, kept scratch) is bit-identical, so
  it is below the traceability line.

## What changed

- `recipes/dense.py`: `BLOCK_ATTN_QKNORM_ROPE = ("qwen3",)`, and `attn_block.prep` for it.
  Fixtures regenerated; they were also stale on main after #139.
- `manifest.{hpp,cpp}`: `AttnBlock::prep`, and `attn_block` parsed on the dense kind when it
  carries one (head dim a multiple of 64 there).
- `core.cpp`:
  - `dense_attention_block` (`attention_prep` + `attention_npu`, no gate);
  - `attention_npu` with an optional gate and the one-chunk scores-then-values order;
  - `step_gemm_block_layer` takes `t_real`, and its tail reads the GEMM outputs as token rows
    from `BlockScratch`.
- `utilities/dense-decode-probe/attn_route_check.py`: the full-model check.
