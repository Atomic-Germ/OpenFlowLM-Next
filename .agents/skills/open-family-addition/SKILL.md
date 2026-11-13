---
name: open-family-addition
description: Adding a new open-kernel family (or size) end to end — recipe vs. design decisions, spec fields and both HF and GGUF builders, oflm-add wiring, frontend routing, kernel builds, and how to verify the result (cmp_layers vs llama.cpp, and what the xres magnitude actually means). Use when a model's architecture is close to but not exactly an existing family, when standing up a second layer geometry in one model (Gemma 4's sliding vs. global layers), or when an install succeeds but the model answers wrongly.
---

# Adding an open-kernel family

Read `open-kernel-pipeline` first (the one-command path), then `open-gguf-direct`
if the source is a GGUF. **This file is what neither covers: how to port a new
family when several of the registry's checkpoints need to change in lockstep.**

The worked example is Gemma 4 12B (OPEN-FAMILY-GEMMA4), ported on `feat/gemma4-direct`
on top of `feat/gguf-direct`. Everything below was built by doing that port, and
includes the traps tripped during it.

## Decide recipe vs. design first

Ask: "is this the nearest family with different constants, or a new layer?"

- Same layer, different constants -> new fields on `ModelSpec` subclassed via a
  new spec builder + catalogue points, no new kernels (nanbeige -> qwen3's
  geometry, granite -> dense).
- A variant layer that still fits a dumped datapath -> one new attention switch
  family-shared (e.g. `ATTN_VNORM`, `ATTN_SCALE`, `ATTN_ROPE_PAIR`, `ATTN_QLOCAL`).
- Two layer types with different geometry in one model, where the kernel must be
  compiled per type (Gemma 4: 40 sliding layers at head dim 256/8 kv + 8 global
  layers at head dim 512/1 kv): the recipe derives two sub-specs and the design
  reads `G4_KIND` from the environment. Two xclbins, two layouts, two position
  tables, per-kernel `kv_row` / `ptab_row` in the manifest.

If the layer truly is not expressible: a new design under `designs/`. That is a
kernel-author project, not this checklist.

## The port checklist

1. **Recipe module** (`open_kernels/recipes/<family>.py`):
   - re-exports `band_bytes`, `chunk_bytes`, `attn_flags`, `probe_env` — `dx.py`
     reads them off the family module, and the export dies
     `AttributeError: module 'recipes.<family>' has no attribute 'chunk_bytes'`
     without them.
   - `G4_KIND`-style selection happens via `os.environ`; `_kind()` raises
     `OpRangeError` for an unknown kind.
   - the recipe raises `OpRangeError` when a family's required knobs are
     missing (`v_norm`, `attn_scale == 1.0`, ...). Refuse loudly, never
     silently degrade.
2. **`ModelSpec` fields** (`recipes/spec.py`): new fields are zero-valued on
   every other family and dropped from `to_dict()` at the default (`_DEFAULTS`
   keyed off `fields(ModelSpec)`), so no shipped spec's hash moves. Remember
   `kernel_hash()` reads `_KERNEL_INDEPENDENT` — new fields are
   part of the kernel hash, which is what you want.
3. **Both builders**: `_family_hf(cfg, real_vocab)` and `_family_gguf(md)`.
   The GGUF header names the geometry (`general.architecture` + the arch's
   keys; per-layer values ride as **arrays** — `sliding_window_pattern`,
   `head_count_kv`). Sanity-check the array against the pattern instead of
   guessing. Refuse the E2B/E4B-style variants by the keys that say so
   (`embedding_length_per_layer_input`, `shared_kv_layers`), not by a size
   guess. Register both in `HF_FAMILIES` / `GGUF_FAMILIES` + `_FAMILY_OF`.
   `hf_model_types()` picks them up for the engine's `hf_config_check`.
4. **`pack.py`**: pack-time folding (e.g. `div_running` — a per-layer scalar,
   applied to following post-norms, in double, one rounding) in NumPy to match
   `pools.cpp` exactly.
5. **`dense.py` / `dx.py` / `attnknobs.py` / `catalogue.py`**:
   - new knobs are typed fields on `DenseGeometry` and live in the shared
     `attn_flags(G, probe_env)` list, **not** in the design — a design-local
     append is how two copies drift apart and the attention gate validates the
     wrong kernel.
   - the new attention-knob tuple enters `CATALOGUE` as its own combination;
     nothing inherits it by shape alone (Gemma 4's sliding layers share
     Gemma 3 12B's attention shape but are not that point — same shape, new
     arithmetic).
6. **Register**: `FAMILIES`, `NOT_IMPLEMENTED`, `families.py`'s dispatch.
   `gen-kernel-spec.py` and `oflm add` both read `FAMILIES` / the
   `GGUF_ARCH_FAMILIES` / `CONFIG_MODEL_FAMILIES` tables — a family present in
   the recipe but absent from the `oflm-add` maps refuses at install with the
   right geometry and "no set serves it".
7. **Frontend wiring** (`src/include/AutoModel/model_families.hpp`): a new
   open-only family is an alias to the adapter that owns its decoder
   (`{"gemma4", SupportedModelFamily::gemma4_12b}`), not a new engine;
   `builtin_backends.cpp` stays untouched because the open engine is the only
   path. Engine route: `_shared_select_open_engine` in `automodel.cpp`.
8. **Build and register**:
   ```bash
   python utilities/gen-kernel-spec.py --repo <hf-repo>     # or --gguf <file>
   python utilities/build-new-spec.py --spec <name>
   ```
   `build-new-spec` runs SPEC -> BUILD -> GATE -> CATALOGUE -> REBUILD ->
   REGISTER (see `open-kernel-pipeline`). Two traps it cost us:
   - the export step builds `--skip-bert` but not the diffusion family;
     `--skip-diffusion` is now passed for LLM specs.
   - `build_design.py` reports `AttributeError` on the family module when a
     re-export is missing — that is the same bug as (1), not a design bug.
9. **Install as a user**:
   ```bash
   export HOME=... OFLM_MODEL_PATH=... OFLM_SETTINGS_PATH=...
   oflm add <repo>            # GGUF-only repos: family derives from the header
   oflm run <tag>
   ```
   If it cannot run, the three usual causes are: family not in the oflm-add
   maps (refuse with `--family`), `model_families.hpp` not updated ("this
   build has no engine for the family"), or the eos list incomplete (below).

## The eos / turn-token trap

`normalize_tokenizer_config` used to fill `eos_token_id` from the token *name*
(`<eos>` -> [1]) and then skip the `generation_config.json`/`config.json` merge
because the array was already a list — so Gemma 4's `<turn|>` (106) never
entered, and every install generated past the turn boundary. The rule is:
**union** every id from all three spellings (tokenizer_config's array,
generation_config, config.json top level + text tower), never pick one.
That merged the gemma-4 12B base into `[1, 50, 106]` and the model stopped
where it should.

## Verifying numerics against llama.cpp

```bash
cmake -S src/open_qwen36 -B build-qwen36 && cmake --build build-qwen36 --target open_qwen36_cli
utilities/ref-dump/dump_ref --model <model.gguf> --ids 1,2,3 --out /tmp/ref/y
OFLM_OPEN_LAYER_NAN=1 OFLM_OPEN_LAYER_DUMP=/tmp/npu \
    build-qwen36/open_qwen36_cli --model <install> --kernels <install>/open_kernels \
    --ids 1,2,3 --prefill-logits
python utilities/quant-compare/cmp_layers.py /tmp/ref/y /tmp/npu
```

`corr` is per-position per-layer; `relL2` is `||got-ref|| / ||ref||`. **`corr`
near 1 but `relL2` huge means one big factor, not divergence** — read the
`|ref|`/`|got|` line from `--per-position` to name it. Gemma 4 is the
canonical case: the recipe folds `layer_output_scale` by having the engine
*carry the residual divided by the cumulative product* (post norms divide by
`P_l = prod_{j<l} s_j`). RMSNorm cannot see the scale (eps makes it visible
only in the subnormals), the tanh softcap preserves argmax, so the answer is
right; the residual magnitude crosses 3e16 while llama.cpp's stays ~10. It is
consistent, not a quant error — but if a layer's `relL2` is large at a
*specific* layer, the bug is there, not in "the quant".

## Things that were almost blamed on the wrong layer

- **The xres explosion was not a quant bug and it is not F16 vs Q4_1.** The
  residual "carried divided by the per-layer scale product" is what divides it;
  the folder is the recipe's `layer_output_scale` fold, and it always looked
  right at bf16 headroom (cap 3.4e38; residual hits ~3e16).
- **The turn-token stop was not the chat template.** The template was fine;
  the eos array from `tokenizer_config.json` was incomplete.
- **"Works on one machine only" is not an NPU property.** Gemma 4's kernels
  compiled and ran on a different box; if anything is machine-specific it is
  the catalogue (unvalidated points), not the ISA.

## Unvalidated points

`OPEN_KERNELS_UNVALIDATED=1` builds a geometry the catalogue has not run. A
release gate does **not** accept that state: the two Gemma 4 attention points
(sliding and global) compiled but were not gated on hardware. An install may
succeed with `OPEN_KERNELS_UNVALIDATED`-built kernels; nobody should call it
validated until `build-new-spec.py` runs its attn gate and the op gates, and
the points enter `CATALOGUE`. The local tuple matching an already-validated
point is **not** sufficient — the knob tuple is separate.
