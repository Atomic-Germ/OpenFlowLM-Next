# Fast prefill for q8 containers, with the weights unchanged

2026-10-06. Status: **plan, for review.** Branch `feat/q8-block-prefill`.

## Why

The published `Atomic-Germ/Qwen3.6-35B-A3B-NPU2` stores its attention and DeltaNet projections
at q8. The block prefill route's GEMM reads q4_1 only, so for this container the recipe emits no
route at all and every prompt prefills one token at a time:

| 656-token prompt, HX 370 | prefill |
|---|---|
| q8, today (token by token) | 84 s |
| q4_1 re-quantized, block route | 14 s |

Re-quantizing to q4_1 isn't an acceptable way to get there: against q8 it flips 86 of 656
next-token choices and diverges at 143 positions (`utilities/quant-compare/README.md`).

There's an exact way. Every q8 code `v` is `16 * hi + lo` with `hi = v >> 4` and `lo = v & 15`, so
a q8 weight is exactly the sum of two q4_1 weights (scale `16d` with min `-128d` on `hi + 8`,
and scale `d` with min 0 on `lo`; both scales are exact in bf16). The route already does this for
one projection, `linear_out` (`out_split`, OPEN-PREFILL-BATCH): it packs the two halves stacked,
runs one GEMM of twice the rows, and adds the halves on the host. This plan uses the same split
for every projection the GEMM route reads.

## What changes

**Recipe** (`open_kernels/recipes/qwen36moe.py` `gemm_route`, shared with Qwen3.5):
- Stop returning no route when `attn` or `linear` is q8. (`shared` at q8 stays refused: the
  family refuses a q8 shared expert outright.)
- For each q8 role, its weight buffer becomes a `from: "pack"` weight: the role's pack ops as
  `std_perm` with `split: "hi"`, then the same ops with `split: "lo"`, stacked. That covers
  `gqkvz_w` (qkv | z), `gqkvg_w` (q | k | v | gate) and `go_w` (o), plus Qwen3.5's dense FFN
  roles. `gout_w` already works this way.
- The step's GEMM is twice the rows (new shapes `gemm_n24576_k2048`, `gemm_n18432_k2048`,
  `gemm_n4096_k4096`). The GEMM core doesn't depend on N, so these are new instruction streams
  on the existing xclbin, with no new hardware context.
- Every split step carries `"split": true`, which generalises `out_split`. `out_split` stays
  readable so existing manifests still load.

**Engine** (`src/open_qwen36/{manifest,core}.cpp`):
- The parser reads `split` per program step, and a split step's weight must be a 2-op hi/lo pack.
- `gemm()`'s callers add the two halves (rows `[0, N)` and `[N, 2N)`) before the existing
  `transpose_parts` / residual add. That is one helper used by every step, replacing the
  out-projection special case.
- Packing reuses `pools::split_q4_1_chunks` as is.

**Cost:**
- Memory: the split copies sit beside the q8 pool, which decode still reads. That's about
  +1.6 GB for the 35B (qkv|z 944 MB, out 315 MB, q|k|v|gate 236 MB, o 105 MB), on an 88 GB
  machine.
- Time: the split GEMMs read twice the q4_1 bytes for those projections. GEMMs were ~8% of a
  block, so the estimate is ~16-18 s for the prompt above, against 84 s today and 14 s for q4_1.

**Not changed:** decode stays at q8 on two contexts (157 ms a token). A q8 one-context decode
image doesn't fit program memory (#116's merged core has 128 B free). Speculative decoding
isn't part of this plan, but its verify pass would reuse the same split, so it would also work
on q8 containers.

## Spec impact

**OPEN-PREFILL-BATCH, modified.** "A spec with `attn`, `linear`, `linear_out` or `shared` at q8
emits none of it" becomes: q8 `attn`, `linear` and `linear_out` are packed as exact q4_1 splits
and the route is emitted; `shared` at q8 is still refused. The acceptance criteria gain the
split emissions and the per-step `split` parse. No new IDs, and nothing is removed.

## Checks

**Unit (`test`):**
- `tests/test_prefill_batch.py`: a spec with `attn` and `linear` at q8 emits a route whose
  `gqkvz_w`, `gqkvg_w` and `go_w` are hi/lo split packs of the right ops, on the doubled
  shapes, with `split: true`. A q4_1 spec's manifest is byte-identical to today's apart from
  the build key.
- `manifest_test.cpp`: `split` parses per step; a split step whose weight isn't a 2-op hi/lo
  pack is refused; an old manifest's `out_split` still loads.
- `split_q4_1_chunks` exactness is already covered (`pools_test.cpp`, `tests/test_qwen35.py`).

**Hardware (`manual`), the 35B q8 container:**
1. Export the kernel set (default, q8): the manifest now carries the route.
2. The 656-token prompt: block route against q8 token by token, logits at every position
   (`cmp_logits.py`). The bar is the route's own spread as measured on q4_1 (median corr
   ~0.9993, flips only at near-ties), not the 0.987 of re-quantizing.
3. All 40 layers on a ~1000-token prompt with `--max-tokens 8`: the same greedy continuation.
4. Prefill time, alternated against the token-by-token route.
5. `oflm-test --llm` through `oflm serve` with the route on.

## Open question

None blocking. If the hardware gate in step 2 shows the split's fp32 sum of two bf16 GEMMs
spreading more than the q4_1 route does, the fallback is a q8-reading GEMM (a new dequant body
in `gemm_q4_prefill`; the core has ~11 KB of program memory free).
