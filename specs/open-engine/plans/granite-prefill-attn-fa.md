# Granite prefill attention on the NPU (attn_fa)

Track P1 of `.claude/plans/granite-overnight.md` (2026-10-15). Branch `perf/granite-prefill-attn-npu`.

## Why

Granite's block prefill takes its attention on the host (OPEN-PREFILL-HOST-ATTN): ~195-240 ms at pp512 and 5-6 s of
~9-14 s at pp2048, growing with the square of the prompt and conflicting with the standing rule that attention belongs on
the NPU. The existing NPU route (OPEN-PREFILL-ATTN, design A: two bf16 GEMMs a kv head with the softmax on the host)
does not fit head dim 64: the score matrix is L / 64 times larger than Q or K, so its fp32 round trip (written by the
array, read and re-written as P by the host, twice with the Q hi / lo split) costs more than the host attention it would
replace.

## Spec impact

- **New requirement OPEN-PREFILL-ATTN-FA**: a dense set's block attention as one causal flash-attention dispatch a
  256-token block (`designs/attn_fa`), scores, mask, softmax and both products on the array; the default for Granite on
  both GEMM routes; `OFLM_OPEN_ATTN_FA=0` keeps the host attention for A/B. Verification: test for the recipe emission
  and the manifest schema (`tests/test_prefill_attn_fa.py`, `manifest_test.cpp`), manual for the kernel and the model.
- **Modified OPEN-PREFILL-HOST-ATTN**: the host attention is now the fallback for a set without `attn_fa` or with it
  switched off.

## Design

`designs/attn_fa/`: 4 groups of 8 cores, a pass is one query head's 256 tokens, each core 32 tokens against every key
chunk of its kv head's window. Query rows are the vector lanes: scores come out as S^T, so the row max, the rescale and
the row sum are lane-wise. Q hi / lo bf16 (no q/k norm), scores in log2 units with an integer running max (exact
power-of-two rescales), a polynomial 2^x to ~2^-12, fp32 O and l. K and V are read straight from the layer's KV cache;
O is written as the out projection's fp32 token rows. One xclbin, a stream per 256 rows of window.

## Steps

1. Kernel + harness (`make_test.py` / `compare.py`, fp64 reference): rel_fro <= 4e-3.
2. Recipe, manifest, engine (`layer_dense_staged`), both GEMM routes; block 0's dispatch starts while the host finishes
   the last block's projection.
3. Gates: `prefill_mode_check.py --arms fast:host,fast:npu` and `bfp16:host,bfp16:npu` at 512 and 2048.
4. CLEAN timing: pp512 / pp1024 / pp2048, both routes, host vs NPU attention, with the stage split.
