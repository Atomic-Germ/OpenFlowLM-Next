# quant-compare

`cmp_logits.py` scores how far one run's per-position logits are from another's (worst / median
correlation, argmax flips and how many were near-ties, top-5 agreement, KL divergence).

## Result 2026-10-06: the 35B's q8 container re-quantized to q4_1

`Atomic-Germ/Qwen3.6-35B-A3B-NPU2` (the same 23.24 GB file as `FastFlowLM/Qwen3.6-35B-A3B-NPU2`)
stores attention, DeltaNet and the DeltaNet out projection at q8. The default export keeps them at
q8, which leaves decode on two hardware contexts and prefill token by token (no block route).
`OPEN_KERNELS_FORCE_Q4_1=1` re-quantizes them to q4_1 at load, which turns on #116's one-context
decode and the block prefill route.

Same 656-token prompt (a summarize request over `open_kernels/README.md`), logits at every
position, HX 370:

| | prefill (logits at every position) | decode |
|---|---|---|
| q8 (default) | 84 s | 156-167 ms/token |
| q4_1 re-quantized, token by token | 51 s | 69-70 ms/token |
| q4_1 re-quantized, block route | 14 s | -- |

| against q8 | logits corr (median / worst) | argmax agree | top-5 identical | KL mean / max |
|---|---|---|---|---|
| q4_1, token by token | 0.9866 / 0.50 | 570 / 656 (86 flips, 43 near-ties) | 277 / 656 | 0.157 / 10.8 nats |
| q4_1, block route | 0.9864 / 0.44 | 570 / 656 (86 flips, 47 near-ties) | 271 / 656 | 0.155 / 10.1 nats |
| (block route vs q4_1 token by token) | 0.9993 / 0.55 | 633 / 656 | 538 / 656 | 0.031 / 8.4 nats |

143 of 656 positions fall below 0.98 correlation, spread through the text. **Quantizing q8
weights a second time, down to q4_1, moves this model far more than the route does.** It is not a
free speedup. A q4_1 (or Q4_K) container quantized once from the original weights has not been
measured.

    open_qwen36_cli --model <35B> --kernels <set> --ids-file ids.txt --max-tokens 1 \
        --prefill-logits --dump-logits <dir>/y [--gemm-block]
    python utilities/quant-compare/cmp_logits.py <q8 dir>/y <q4_1 dir>/y
