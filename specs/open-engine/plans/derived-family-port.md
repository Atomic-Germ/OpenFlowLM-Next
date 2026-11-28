# Deriving a family port instead of hand-writing one

Finding from comparing `open_kernels/recipes/*.py` here with the implemented
families in `../cyronius-open-kernels` (flint, hyv3, nemotronh, xing, gptoss,
dflash2). The open-kernel families are **mostly pure functions of ModelSpec**;
the apparent "rigor" that forces a human to hand-write a family is largely
legacy closed-ecosystem protection, not a technical limit.

## What a recipe actually is

Every family module exposes the same surface:

```
recipe(spec, max_ctx)  layout(spec, max_ctx)  pack_plan(spec)
programs(spec)  builds(spec)  hf_config_check(spec)
manifest_layout(spec, max_ctx)  attn_flags(G)  KERNEL_SOURCES  GEN_KERNELS
```

For flint (635 lines) the ONLY inputs are `ModelSpec` fields:
`hidden, num_experts, experts_per_tok, moe_intermediate,
shared_expert_intermediate, num_heads, num_kv_heads, head_dim, rotary_dim,
qk_norm, attn_gate, layer_types, rope_theta, ...`. All the geometry, layout,
pool, pack ops, step program and manifest are arithmetic on those.

## Constants that look like "port work" but are derivable

| constant | values seen | source of truth |
|---|---|---|
| `CH` / `CHUNK` | 5120 (q4_1), 4608 (q4_0), 5120 (q8) | `spec.quant` + pool law (`pools.cpp`) |
| `N_CORES=8`, `BAND_ROWS=64`, `ELEM=4096`, `STACK=0x1800`, `LM_BAND_ROWS=128` | identical in all families | NPU hardware, universal |
| `TOPK_MAX`, `SLOTS`, `ATTN_CORES`, `CPS` | vary | spec head/expert counts, `attnknobs.attn_cores()` |
| `KERNEL_SOURCES`, `GEN_KERNELS` | per design | family's layer kinds (dense / MoE / conv / attention variant) |

`gemv_q4.h` and the q4 band law are shared by flint and hyv3 verbatim. The two
`gen_kernels.py` files are ~90% identical template code parameterized by the
recipe's `Geometry`; they differ only in the routing arithmetic (slot weights,
pair law), itself driven by spec fields.

## The pathway (no new math required for a "similar family")

1. **derive constants** from `ModelSpec` + quant + hardware (table above).
2. **generate the recipe module** from one of a few shape templates:
   dense / MoE-with-router / hybrid-conv / attention-variant. The template
   is chosen by the same capability check `build-new-family` already uses.
3. **generate `gen_kernels.py`** from the family's Geometry + a kernel-source
   pattern (flint/hyv3/nemotronh/xing/gptoss each show a distinct pattern).
4. **generate catalogue points** by running `build-new-spec --dry-run`, which
   already lists every missing point; `op-gate.py` runs the fixtures that
   exist and defers the rest (whole-layer path). This is already scripted.
5. **verify** with the numeric gates (`cmp_layers` vs llama.cpp) — a failed
   gate is the only honest signal that the family needed real new arithmetic.

Steps 1–4 are mechanizable. Step 5 needs hardware and a reference model, but
its invocation is already chained in `build-new-family --cli/--ref-model`.

## First tractable slice

Generate the **recipe module** for a family whose shape matches an existing
template but whose spec values differ, reusing that family's design directory
and `gen_kernels.py` untouched (the flint/hyv3 relationship). Concretely: a
new spec field set that fits `flint`'s `_check()` but is not flint — generate a
recipe module by substituting spec-derived constants, then run
`build-new-spec` (which derives geometry/layout/pack/programs/catalogue/gates
from it). This is "register-alias" one level up: alias the *geometry* onto a
recipe template rather than just the *name* onto a builder.

Require inputs that cannot be derived but are possible to look up: the design
directory (`--design`) and the kernel-source pattern (`--pattern`) when the
family introduces a new layer type; the spec target repo for numeric gates.
