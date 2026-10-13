# Let the user pick the q8 35B's prefill route: `--prefill-mode fast|lean`

## Why

#181 moved the all-q8 35B's block route from #172's q4_1 split to bf16 GEMMs. That took
prefill at 1024 tokens from about 116 to about 145 tok/s, but the route's weights grew from
1.49 to 2.62 GiB. The extra memory doesn't matter on an 88 GB machine and may on a 32 GB one,
so the user picks the trade-off when the model loads.

## Requirement changes

- **OPEN-PREFILL-MODE (new).**
  - A set may carry `gemm_block_variants` beside each layer type's `gemm_block`.
  - The all-q8 35B set carries the bf16 route as `gemm_block` and the split route as `lean`.
  - `oflm run|serve|bench --prefill-mode fast|lean` sets `OFLM_OPEN_PREFILL_MODE`;
    `open_qwen36_cli --prefill-mode` does the same.
  - The engine swaps the chosen route in before it loads kernels and packs weights, so only
    that route's xclbins, weights and globals exist.
  - `lean` on a set with one route logs that it changes nothing.
- **OPEN-PREFILL-BATCH (modified).** The all-q8 set ships both routes.
  `OPEN_KERNELS_Q8_GEMM` is gone, because the runtime mode replaces it.
- **OPEN-MANIFEST (modified).** The parser reads `gemm_block_variants` and checks each variant
  as it checks `gemm_block`. It also refuses three cases:
  - a variant without a `gemm_block`;
  - a variant whose `t` or `kind` differs from `gemm_block`'s;
  - layer types that carry different variant names.

  An engine that predates the field ignores it. `manifest_version` stays at 3.

## Design notes

- **The mode travels through the environment.** `LM_Config` has to keep its layout across
  the OFLM_DLL boundary, and adding a parameter to every `load_model` would touch every model
  class.
- **`Manifest::select_prefill_route` drops the unused route's globals.** A global is dropped
  when only the unused route names it: the split's y buffers under fast, about 48 MB, and the
  bf16 route's under lean.
- **`--prefill-mode` is refused outside `run`, `serve` and `bench`, and for any value other
  than fast or lean.** A flag that is accepted and then ignored reads as one that took effect.

## Results (2026-10-08, published 35B, HX PRO 370)

- **Logits match byte for byte.** The comparison covers the last prompt position and two
  decode steps after 1024 tokens.
  - Fast matches the #181 set and lean matches the #172 set.
  - Lean on the #172 set logs the one-route line and also matches.
- **Prefill at 1024 tokens, 10 interleaved runs per mode:**

  | mode | median | tok/s |
  |---|---|---|
  | fast | 7078 ms | 144.7 |
  | lean | 8799 ms | 116.4 |

  - Lean's throughput is about 20% lower, not the ~15% the plan quoted from #181's earlier runs.
  - Single runs swing about ±5%.
- **Memory, peak private bytes:**

  | | peak private bytes |
  |---|---|
  | fast | 24.70–24.71 GiB |
  | lean | 23.74–23.75 GiB |
  | #172's set | 23.74 GiB |

  - System commit differs by 1.10–1.14 GiB between the modes.
  - The private-bytes difference is smaller, 0.96 GiB.
- **`oflm-test --llm` through `oflm serve`** passed 5 of 5 under `--prefill-mode lean` and
  under the default mode. Turns of 169 to 953 tokens took the route.
- **Disk:** the set is 13 MB, up from 11 MB with one route.
