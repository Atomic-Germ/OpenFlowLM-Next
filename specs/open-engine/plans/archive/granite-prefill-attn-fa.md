# Granite prefill attention on the NPU (attn_fa)

Track P1 of `.claude/plans/granite-overnight.md` (2026-10-15). Branch `perf/granite-prefill-attn-npu`. Done; the
durable result is OPEN-PREFILL-ATTN-FA in `specs/open-engine/spec.md`.

## Why

Granite's block prefill took its attention on the host (OPEN-PREFILL-HOST-ATTN): ~200-270 ms at pp512 and 4-8 s at
pp2048, growing with the square of the prompt and conflicting with the standing rule that attention belongs on the NPU.
The existing NPU route (OPEN-PREFILL-ATTN, two bf16 GEMMs a kv head with the softmax on the host) does not fit head dim
64: the fp32 score matrix is L / 64 times larger than Q or K, and its round trip (written by the array, softmaxed and
re-written as P by the host, twice with the Q hi / lo split) costs more than the host attention. Not built.

## Spec impact

- New OPEN-PREFILL-ATTN-FA (test for the recipe and the manifest schema; manual for the kernel and the model).
- OPEN-PREFILL-HOST-ATTN: the host attention is now the fallback for a set without `attn_fa` or with it off.
- OPEN-PREFILL-BFP16: its "host attention only" sentence now points at OPEN-PREFILL-ATTN-FA.

## What was built, in order, and what each round found

1. dit_fa's structure at head dim 64 (emulated bf16 `mmul<4,8,8>`, 8x8-block layouts from the memtile): correct first
   time (rel_fro 1.13e-3) but 1.73 ms at L 512; the hi / lo operands' expansions spilled to the stack every iteration.
2. Query rows as the vector lanes (S^T), element-wise MACs, no memtile transforms: wrong (rel_fro 1.25) until bf16
   `broadcast_elem` was understood to broadcast a 32-bit pair; values doubled with `interleave_zip` first.
3. Ablations at L 512 (CLEAN): no compute 0.4-0.6 ms, softmax 0.62, P.V 0.34, Q.K^T 0.17. The softmax was a serial
   chain; four interleaved keys and a shorter polynomial: 1.61 -> 1.42 ms, after the stack went to 8 KB (a 5 KB frame
   at a 4 KB stack first gave NaN in ~5 % of rows, then a hang). Four passes in flight hang; contiguous K / V reads change
   nothing.
4. Engine: Q packed as Q^T per 32-token tile (a row-major writer gave ppl 190022), tile-parallel (token-parallel strided
   writes cost ~180 ms of host time at pp512, now ~50).

## Results

See OPEN-PREFILL-ATTN-FA. pp2048: bf16 route 183 -> 247 tok/s, bfp16 route 269 -> 480. pp512: 240 -> 234 and
542 -> 500 (two context switches a layer). The bf16 route meets every accuracy threshold; the bfp16 route at 512 fails
two on one prompt each (a flip at margin 0.59, one continuation diverging at token 35) while matching the bfp16 route
with host attention on corr, KL and perplexity against the default route.

## Not done

- A context shared with the GEMMs (pp512's loss), the kernel at its MAC bound (~1.7x), Q without the hi / lo split, RoPE
  and Q packing on the NPU, windows past 2048 on hardware (the recipe emits them to 4096; the measured set stopped at 2048).
