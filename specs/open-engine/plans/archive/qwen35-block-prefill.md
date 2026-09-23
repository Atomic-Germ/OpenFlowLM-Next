# Batched prefill for Qwen3.5 (issue #109) -- ARCHIVED 2026-09-23, implemented

## The problem

A Qwen3.5 model (the `qwen35` family: Qwen3.5-0.8B/2B/4B/9B, the Qwen3.8-Distilled
sizes, Ornith1.5-9B) prefills one token at a time. `recipes/qwen35.py` has no
`gemm_route()`, so its kernel set carries no `gemm_block` program and the engine logs
`block prefill route: T = 0`. bvlob's 9B: 8.5 tok/s prefill, 113 s to first token on a
~960-token prompt.

This isn't an engine limitation. A Qwen3.5 layer is a Qwen3.6-35B layer with the MoE
block replaced by a SiLU-gated dense FFN (the recipe's own docstring). The 35B's route
already runs the rest of the layer: the DeltaNet (`linear`) and full-attention (`full`)
halves, with host stages that are tested against the numpy reference. The dense FFN it
needs is also already there: the 35B's **shared expert** is a SiLU-gated dense FFN,
run as two GEMMs (up|gate, then down) by `Core::shared_expert_block`. The only
difference is the shared expert's sigmoid gate, which the dense FFN doesn't have.

So the fix is mostly composition: the 35B's `linear` / `full` route, with the FFN tail
switched from "router + shared expert + routed experts" to "dense FFN".

## Also found

1. **`OFLM_OPEN_GEMM_BLOCK=1` on a set with no route skips the prompt.**
   `engine.cpp` `Engine::prefill`: inside the `gemm_block_env == "1"` branch, when
   neither `layer_major_ok()` nor `GT > 0` holds, it `return logits_view()` without
   calling `step()`. The comment above says such a set "runs exactly the sequential
   path". Today anyone who sets the flag on a qwen35 model gets an unprocessed prompt,
   which is exactly what bvlob would have tried next. It's a bug against
   `OPEN-PREFILL-BATCH`, fixed by falling through to the sequential loop. Separate commit.
2. **The route is opt-in.** Even after this lands, bvlob sees no change without
   `OFLM_OPEN_GEMM_BLOCK=1`. `OPEN-PREFILL-BATCH` says "off by default so every existing
   measurement is unaffected". Flipping that default is a separate decision (open
   question 1), out of scope here.

## Changes

### Recipe (`open_kernels/recipes/`)

- `qwen36moe.gemm_route(spec, ffn="moe")` gains the `ffn` parameter the rest of that
  module already takes. With `ffn="dense"`:
  - `linear` / `full` keep `kind`, `program` (the fused input projection, then out/o),
    their weights and the attention/DeltaNet fields, byte-for-byte the 35B's.
  - Instead of `moe_kernel`, `moe_args`, `moe_batch`, `shared_*`, the type carries
    `ffn_program`: `[run(2*ff, hid, "gffn_ug_w"), run(hid, ff, "gffn_down_w")]`, with
    `ff` = `spec.intermediate`. The weights are pool ops up+gate (contiguous at pool
    offset 0, up first, which is the layout `shared_expert_block` reads) and down.
  - No `mx_*` / `mb_s*` contexts, kernels or builds. The GEMM context and the
    `attn_block` (`ag_*`) builds are emitted as for the 35B. The 9B's
    16/4 heads give `AG_M` 1024 against the 35B's 2048, which is new builds but not a new design.
  - The 35B's own emission must not move: `ffn="moe"` is the default and its manifest
    fixture is the regression check.
- `qwen35.programs()` merges the route the way `qwen36moe.programs()` does.
- A shape that isn't a multiple of 256 returns `None` (a sequential set) rather than
  raising, so one odd size can't break a family export. All four catalogue sizes are
  multiples of 256 today; the test pins that each of them gets a route.

GEMM shapes at the 9B (T = 256): `n12288_k4096` (qkv|z), `n4096_k4096` (out, o),
`n10240_k4096` (q|k|v|gate), `n24576_k4096` (up|gate), `n4096_k12288` (down).
`N = 24576` is twice the largest N the GEMM has run (12288). If the design refuses it,
the fallback is a three-step FFN (gate, up, down), as the dense route does. The
engine's host tail reads either.

### Engine (`src/open_qwen36/`)

- **manifest.cpp**: a `linear` / `full` route carries **either** the MoE tail
  (`moe_kernel` + `shared_program`, as today, requiring `layout.moe`) **or** an
  `ffn_program` of two 3-arg runs naming declared weights. Both present, or neither,
  is refused by name. `manifest.hpp` gets `ffn_program` on `GemmBlockProgram`.
- **core.cpp `load_weights`**: `router` / `sgw` are read only for the MoE tail.
  DeltaNet alpha/beta: the 35B's container stores `[hidden, heads]`, which the host
  reads directly. Qwen3.5 stores q8 plus a bf16 `[heads, hidden]` copy, which the
  consts plan names `ssm_alpha_proj.bf16.weight`. When the consts op is a
  `transpose`, the host transposes it to `[hidden, heads]` with `lanes = heads`.
  `deltanet_block` only needs `lanes >= value_heads`, so no 32-lane padding is
  needed on the host.
- **core.cpp `block_layer_linear` / `block_layer_full`**: after the post-attention
  norm, a dense-FFN type runs `ffn_block` (up|gate GEMM, `silu(g)*u`, down GEMM) and
  writes `xres = res + down` for the block's rows. A MoE type runs the router and the
  shared expert as today. `shared_expert_block` becomes `ffn_block(..., gate)` with a
  null gate meaning ungated, so both tails share one code path.
- **`block_layer_moe`** is a no-op for a dense-FFN type (it already wrote `xres`). So
  `step_block_moe` and `step_gemm_prompt` (the layer-major schedule) run unchanged.
  Layer-major stays on for qwen35. It isn't there for the experts here, but it still
  saves the per-block DeltaNet state sync and KV pull-back that it saves on the 35B.
- `layer_major_ok()` needs no change (kinds are still `linear` / `full`).
- **engine.cpp**: the fall-through fix above.

The route keeps a second copy of every projection it runs as a GEMM weight buffer
(`build_weights`), as the dense route already does for Qwen3-8B. For the 9B that's
roughly +4.7 GB resident. This is worth saying in the PR, and it's the reason the flag
default matters (open question 1).

## Spec impact

Modified requirements, no new ones:

- **OPEN-PREFILL-BATCH**: `Applies to` adds `recipes/qwen35.py`. The body gains the
  clause that a `linear` / `full` type carries either the MoE tail or a dense
  `ffn_program` (up|gate then down, SiLU on the host, ungated, residual added). New
  unit acceptance criteria are listed below. The engine.cpp fix restores behaviour the
  requirement already states.
- **OPEN-MANIFEST**: the schema clause for `ffn_program` and the either/or rule.
- **OPEN-FAMILY-QWEN35**: one sentence pointing at the route and its result.

## Verification

`test` (deterministic, silent when wrong, cheap):

- `tests/test_qwen35.py`: the 9B's emission. It checks the five shapes and their pool
  and consts op indices, `ffn_program` over pool ops 0-1 then 2, no `mx`/`mb`
  entries, the `ag` builds at `AG_M` 1024, and that a spec with `linear_out` or
  `ffn` at q8 emits no route. The same test also checks that all four catalogue
  sizes emit a route.
- `tests/test_prefill_batch.py`: the 35B's emission is unchanged (existing
  assertions, plus the `ffn="moe"` default).
- `manifest_test.cpp`: a qwen35 fixture parses without `layout.moe`, and both-tails
  and neither-tail are refused by name.
- `block_host_test.cpp`: the alpha/beta transpose helper, and `ffn_block`'s host math
  ungated against a plain loop. This only applies if they land as free functions in
  `block_host`; if they stay inline in `Core`, they're covered by the manual gates.

`manual` (needs the NPU; `Qwen3.8-Distilled-9B-NPU2` is local, and it's the same shape
as Ornith1.5-9B):

1. Export the 9B kernel set; each new GEMM shape through the harness
   (`make_test.py --shape nN_kK --tokens 256` → `compare.py` PASS).
2. `open_qwen36_cli --layers 4 --prefill-logits --dump-logits`, with and without
   `--gemm-block`, on a prompt of more than 256 tokens (so a full layer and a second
   block are both exercised). Pass: argmax and top-5 equal except documented near
   ties (margin < 0.05), and corr > 0.999 per position.
3. All 32 layers, about 1000 and 2582 tokens, `--max-tokens 8`: the same greedy
   continuation (the decode-after-block-prefill gate: the route writes the DeltaNet
   state and KV rows that decode reads), with TTFT recorded both ways.
4. The same prompts, layer-major against `--block-major`: byte-identical logits (there
   are no experts, so nothing should differ).
5. `oflm-test --llm` through `oflm serve` with `OFLM_OPEN_GEMM_BLOCK=1`. Also check
   that the serve path with the flag on and an old set (no route) now prefills
   sequentially: the engine.cpp fix.
6. Repeat 1-3 on `Qwen3.5-0.8B-NPU2` (hidden 1024, the smallest geometry).

## Risks

- **`N = 24576` in `gemm_q4_prefill`**: untried. The fallback is described above.
- **Host memory for the dense FFN at the 9B**: `gemm_y_n24576` is 25 MB and the host
  `ug` buffer is 25 MB a block. Both are fine.
- **Numerics**: the route's bf16 GEMM flips near-tied argmaxes on the 35B, and the
  same is expected here. The 9B's `ssm_out_proj` is requantised q8 → q4_1 on both
  paths, so the two paths agree with each other; neither matches the q8 reference
  exactly.
- **Conflicts with #101**: it touches `step_gemm_block_layer` and the `dense` manifest
  branch; this plan touches the `linear` / `full` branch and functions. They should
  merge cleanly, but whichever lands second re-checks `manifest_test.cpp`.

## Decisions (2026-09-23)

1. **Default on.** `OFLM_OPEN_GEMM_BLOCK` now defaults on; `=0` turns it off. That makes
   every family whose kernel set has a route take it by default, including Phi-4-mini, whose
   route hangs on main's `dx_attn` (#100). So this branch is stacked on #101, which fixes
   that, and has to merge after it.
2. **One PR** for the route, the fall-through fix and the default, tied to #109.

## Found during implementation

- **The published 9B stores `ssm_out_proj` at q8, and the engine streams it at q8**
  (the spec derived from the model says `linear_out=q8`; the checked-in `qwen35-9b.json`
  says q4_1, which is why the unit tests passed while a real export emitted no route).
  The GEMM reads q4_1 only. A first cut packed a re-quantised q4_1 copy; against the fp64
  replica at full depth it scored mean logits corr 0.890 / min 0.44 / top-5 2 of 12 on hard
  positions (markdown table separators), where the sequential path scores 0.99998 -- the
  same loss OPEN-QUANT-Q8 measured when it made q8 native. With identical q4_1 weights on
  both paths the route matched the sequential path to corr 1.00000 at one layer, so the
  route's own arithmetic was never the problem.
  **Fix:** every q8 code v = 16 hi + lo, so the weight is the exact sum of two q4_1
  readings (d = 16 scale, m = -128 scale, nibble hi + 8; d = scale, m = 0, nibble lo).
  `std_perm` gained `split: hi | lo`, the route packs both halves stacked (`from: "pack"`),
  one `gemm_n8192_k4096` runs them, the host adds (`out_split`). Result: full depth vs the
  fp64 replica mean 0.99984 / min 0.99885 / argmax and top-5 12 of 12.
- The attention-GEMM builds bake `AG_M`, and the 9B's 4 query heads per kv head give 1024
  against the 35B's 2048, so those builds get an `_m1024` directory suffix (the 35B's names
  are unchanged).
- The 35B's manifest is byte-identical to before apart from the build key.

## Open questions (resolved)

Both were decided on 2026-09-23 -- see "Decisions" above: the route defaults on, and the
route, the fall-through fix and the default ship as one PR tied to #109. Results are in
`spec.md` under OPEN-PREFILL-BATCH ("Result 2026-09-23 (Qwen3.5, #109)").
