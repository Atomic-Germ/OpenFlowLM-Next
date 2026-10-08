# q8 projections on the block route: one bf16 GEMM context, not a q4_1 split

## Why

#172 runs every q8 projection on the route as its exact q4_1 split: [hi | lo] stacked into one
GEMM of twice the rows, the halves summed on the host. Its plan assumed GEMMs were ~8 % of a
block. On the q8 35B at 1024 tokens (2026-10-07, `OFLM_OPEN_DISPATCH_LOG=1`, prefill 8354 ms)
the split GEMMs are 2832 ms (34 %): `gemm_n24576_k2048` 1601, `gemm_n4096_k4096` 823,
`gemm_n18432_k2048` 408. That is more than the expert stage (2358 ms). Doubling the rows
doubles the matmul, the dequant and the activation refetch together.

## Requirement changes

- **OPEN-PREFILL-BATCH (modified).** On a MoE container whose route projections are all q8 (the
  35B), every route GEMM reads a bf16 pool on one hardware context, `gemmb`
  (`gemm_q4_prefill` built with `GQP_FMT=bf16`): the q8 projections at their real row count,
  and the shared expert's two GEMMs. Each weight is rounded to bf16 once, at load. A container
  with only some roles at q8 keeps the exact q4_1 split on `gemm`. `OPEN_KERNELS_Q8_GEMM=split`
  forces the split everywhere, for A/B.
  Acceptance (manual, on the 35B):
  - every bf16 shape PASSes the harness (rel_fro ≤ 5e-3);
  - 1024 positions against the q8 sequential path, no worse than the split route on the same
    prompt (corr, argmax flips, near ties);
  - prefill faster than the split route.
- **OPEN-PACK-PLAN (modified).** One pack op, `bf16_gemm`, byte-identical to its NumPy twin
  (`recipes/pack.py`), verified by `test` in `pools_test`. Source: `nch` q8 or q4_1 chunks from
  `chunk0`. Output: bf16(code × scale) or bf16(n·d + m), band-major 64 × 64 elements in mm.cc's
  A-block order.
- **OPEN-MANIFEST (modified).** A set with `bf16_gemm` says `manifest_version` 3; an engine that
  reads 1–2 refuses it by name. A packed weight is one format. Buffer sizing goes through
  `pools::op_bytes`.

## Results so far

**Phase 1, harness** (`designs/gemm_q4_prefill`, T = 256, min of the warm runs, same tree and
toolchain for every arm):

| real shape | q8 | bf16 | q4_1 split |
|---|---|---|---|
| n2048 k4096 (`ssm_out`) | 4.94 ms (0.54×) | 3.98 ms (0.44×) | 9.11 ms |
| n12288 k2048 (qkv\|z) | 15.34 ms (0.54×) | 12.31 ms (0.43×) | 28.65 ms |
| n9216 k2048 (q\|k\|v\|gate) | 11.46 ms (0.53×) | 9.32 ms (0.43×) | 21.48 ms |

- Every arm PASSes: rel_fro 1.63e-3 against fp64, 5.9e-7 against the bf16-rounded weights.
  The q8 and bf16 outputs are bit-identical.
- The harness runs about 2.2× slower than the same instructions inside the engine. #172's own
  `gemm_n24576_k2048` insts gave 29 ms here against 12.6 ms in its dispatch log, with
  byte-identical output to this tree's q4_1 build. So the ratios hold and the absolutes don't.

**L1.** Reading decode's q8 band-law pool in place needs a 20 KB band, double-buffered, and the
core has ~4 KB spare. The q8 format repacks into 8704-byte elements instead: 64 rows × 128 K
with codes already in A-block order, so the on-core dequant is convert, scale and round, with no
transpose. Its own pool is 17/16 B a weight, half the split's.

## In the engine (2026-10-07, 1024 tokens, `OFLM_OPEN_DISPATCH_LOG=1`, min per call)

| kernel | split | q8 | bf16 |
|---|---|---|---|
| qkv\|z | 12.57 ms (n24576) | 9.17 ms | 7.87 ms |
| `ssm_out` / `o_proj` | 4.22 ms (n4096) | 2.37 ms | 2.06 ms |
| q\|k\|v\|gate | 9.44 ms (n18432) | 7.53 ms | 6.55 ms |

- In-engine the GEMM behaves close to core-bound, not stream-bound. That favours the format with
  no on-core dequant.
- The q8 and bf16 routes give bit-identical logits end to end, as designed (the same
  once-rounded weights).
- **A second GEMM context gives much of it back.** The shared expert's q4_1 GEMMs stay on `gemm`,
  so every block switched gemm8 → gemm → gemm8. `gemm_n1024_k2048` went 0.68 → 3.14 ms a call
  (+444 ms a prompt), because the recipe's ~2.5 ms context change lands on it.

**Decision.** The route of a MoE container whose projections are all q8 (the 35B) runs every
GEMM on one bf16 context, `gemmb`. That includes the shared expert, whose q4_1 weights are
dequantized to bf16 at load (n·d + m rounded once, ~250 MB on the 35B). Weights are rounded to
bf16 once at pack time. That is 2 B a weight against the split's 1.25, so the route's weights go
1.49 → 2.62 GiB on the 35B (+1.13 GiB, from the two sets' manifests).

The on-core q8 format is dropped: slower than bf16, and it needs a second context. Its numbers
stay above as the record. A container with only some roles at q8 (Qwen3.5: `ssm_out` alone)
keeps the split, which stays in one q4_1 context.

So the requirement changes above read: **one pack op, `bf16_gemm`**, from q8 or q4_1 source
chunks. Context `gemmb`, kernels `gemmb_nN_kK` (`GQP_FMT=bf16`). `OPEN_KERNELS_Q8_GEMM` =
`bf16` (default) | `split`.

## Results (2026-10-07, published 35B, alternated processes, `--repeat 4`, warm reps)

| | split (#172) | bf16 route |
|---|---|---|
| prefill, 1024 tokens | 8846 ms (115.8 tok/s) | 7479 ms (136.9 tok/s) |
| GEMM column, 1024 | 3421 ms | 2002 ms |
| GEMM column, 2582 (two rounds) | 11.2 / 11.8 s | 6.9 / 6.2 s |
| prefill, 2582 (two rounds) | 24.7 / 26.0 s | 23.4 / 19.8 s |
| vs sequential, 1024 positions: flips (all near ties) | 29 | 27 |
| worst / median corr | 0.9537 / 0.99946 | 0.9649 / 0.99941 |
| KL mean / max (nats) | 0.00308 / 0.226 | 0.00294 / 0.066 |
| route weights | 1.49 GiB | 2.62 GiB |

- Every bf16 rep beat every split rep in each round.
- `oflm-test --llm` through `oflm serve` with `OFLM_OPEN_KERNELS_DIR` at the set: PASS 5 of 5
  (the 169- and 618-token turns took the route).
- At 2582 the expert stage, which this does not touch, moved 2.2 s between rounds.
- The first accuracy attempt met six NPU context errors (`lx1` decode timeouts, on #172's set
  too) and a block of zero logits. The rerun had none, and is what is quoted here.

## Notes for later

- The 35B's shared expert is stored at q8, and both the sequential path and this route read
  it re-quantized to q4_1 (`via: q4_1`). Reading it exactly, in both, is its own change.
- The harness times every GEMM about 2.2× slower than the engine does, from the same
  instructions with byte-identical output. Compare ratios there, not absolutes.
