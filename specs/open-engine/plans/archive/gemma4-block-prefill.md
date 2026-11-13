# Plan: Gemma 4 block prefill (item H)

**Status:** implemented 2026-10-20 (E2B, E4B gated; the 12B gate and the NPU attention default are open, see spec.md OPEN-PREFILL-G4). **Branch:** `perf/gemma4-block-prefill`, stacked on `perf/gemma4-one-context`
with item A's commits (`feat/gguf-block-prefill`) cherry-picked.

**Spec impact:**
- **New:** `OPEN-PREFILL-G4` -- the Gemma 4 block prefill route (E2B / E4B first, then the 12B).
- **Modified:** `OPEN-PREFILL-BATCHED` (the known-gaps paragraph loses Gemma 4), `OPEN-FAMILY-GEMMA4`
  (cross-reference only).
- No requirement removed. The decode kernel sets, the pool and the position tables do not change.

## Why

Gemma 4 prefills one token per full pass, so prefill tok/s equals decode tok/s. Item A gave dense GGUF models a
block route, but its kind `dense` knows one layer shape. Gemma 4 has four layer kinds, KV sharing, per-layer
inputs, two head widths and a scalar after each layer.

## Design

One new route kind, `gemma4`, in each layer type's `gemm_block`:

- **GEMMs:** the existing `gemm_q4_prefill` (T = 256), unchanged. Gemma 4's pool is the bf16-scale 5120-byte chunk
  (`std_perm_gguf` narrows at pack time), which that GEMM already reads, so no f32 twin. Seven steps per layer in a
  fixed order: input projection (q|k|v on an own-cache layer, q alone on a shared one), o, gate, up, down, and the
  per-layer input's gate (256 x hidden) and projection (hidden x 256), whose packed copies are already in each pool.
  All shapes are multiples of 256 and run on one `gemm` hardware context; the block never leaves it.
- **Attention on the host**, in `block_host.cpp`: q/k RMS norm with weight, v RMS norm without, the kind's RoPE
  (full-head pairs on sliding layers, the first quarter of pairs spanning the head on global ones), unscaled scores,
  causal plus the 512-row window on sliding layers, K / V cached as bf16 in the decode layout `[k | v]`. A shared
  layer reads its source layer's rows (the same buffer). The cost grows with the context on the 7 (E2B) / 7 (E4B)
  global layers and is bounded by the window on the rest; the NPU products for the global layers are the next step
  (see results).
- **Per-layer inputs in batch:** the model projection over the block is one host f32 matrix product (the weights are
  bf16, not 4-bit, and are never re-quantized); the gate and projection are NPU GEMMs; the GELU, norm, residual and
  layer scalar are host loops over the block.
- **Engine:** `step_gemm_block` gains the `gemma4` kind. Logits come from the last real token through the unchanged
  `ln` / `lm` tail.

## Requirements (to merge into spec.md)

- `OPEN-PREFILL-G4` (test): the recipe emits a `gemma4` route per layer type -- 7 steps own / shared, weights as
  contiguous pool runs of exactly N x K x 5/8 bytes, a build per shape, the same entries in the manifest's `gguf`
  view -- and the decode sections are byte-identical to the base commit's (`tests/test_gemma4_block_prefill.py`).
- `OPEN-PREFILL-G4` (test): the host attention against an independent naive double-precision implementation on
  random input, per kind (`block_host_test.cpp`).
- `OPEN-PREFILL-G4` (manual): block route against the one-token route on a few hundred tokens, `oflm-test --llm`,
  and `OPEN-PREFILL-BATCHED`'s procedure; E2B and E4B.

## Order

1. Recipe + Python test; 2. manifest parse + `manifest_test`; 3. host attention + test; 4. `Core` route; 5. build
the GEMM shapes natively; 6. CLI comparison, serve, `oflm-test`; 7. measure; 8. the global layers on the NPU
(only if it pays); 9. the 12B; 10. spec merge, archive.
