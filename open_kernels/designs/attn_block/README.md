# attn_block: block attention as two bf16 GEMMs (prefill step B, design A)

Plan: `.claude/plans/prefill-step-b-attention.md`. Requirement: `OPEN-PREFILL-ATTN`.

`attn_gemm.py` is a thin module over the bf16 whole-array GEMM the embedding models
already run on (`npu_offload/gemm_rtp/gemm_pretiled.py`), built with its runtime loop
bounds so one xclbin carries an instruction stream per window length. `make_test.py`
writes one kv head's two products at the 35B's shape -- 8 query heads x 256 tokens
against L cached rows, head dim 256 -- with fp64 references; `compare.py` gates at
rel_fro 5e-3.

## Measured 2026-09-12 (L = 2048, Strix, clean box)

```
python make_test.py --L 2048
AG_M=2048 AG_K=256  AG_N=2048 python build_design.py designs/attn_block/attn_gemm.py designs/attn_block/build_s2048
AG_M=2048 AG_K=2048 AG_N=256  python build_design.py designs/attn_block/attn_gemm.py designs/attn_block/build_pv2048
../../harness/out/run_kernel.exe run_s2048.cfg  && python compare.py s2048
../../harness/out/run_kernel.exe run_pv2048.cfg && python compare.py pv2048
```

| product | shape (M x K x N) | GFLOP | per dispatch | rel_fro |
|---|---|---|---|---|
| S = Q K^T | 2048 x 256 x 2048 | 2.15 | 0.95 ms | 1.1e-7 |
| O = P V | 2048 x 2048 x 256 | 2.15 | 0.96 ms | 6.9e-7 |

2.2 TFLOPS either way (the first run of a set is ~4 ms, the context's warm-up). The two
builds' `final.xclbin` differ in 72 bytes -- the UUID and timestamp -- and only the
instruction streams differ, so every L is a stream over one context.

Per 256-token block at 2048 tokens of window: 2 kv heads x 2 products x 10 layers =
40 dispatches, about 40 ms, against ~2.5 s for the same products in
`host::attention_block`. The host keeps the norms, RoPE, the cache write, the causal
mask and the row softmax (the softmax is ~8 M exponentials per layer at L 2048).
