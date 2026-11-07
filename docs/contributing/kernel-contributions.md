# Kernel Contributions

> "The code that matters." -- Keep it simple, clear, and focused.

This document covers contributing **open kernel sets** (AIE xclbins + `manifest.json`)
for OpenFlowLM's NPU engine.

**Read this first:** the authoritative description of the path is
`.agents/skills/open-kernel-pipeline/SKILL.md` (the one-command pipeline) and
`.agents/skills/open-kernel-families/SKILL.md` (the family recipes and their traps).
This page is the short tour; the skills are the working manual.

---

## Overview

OpenFlowLM runs models on AMD's XDNA2 NPUs. `open_kernels/` holds the **design sources**
that are compiled into `.xclbin` files and shipped as a `manifest.json` the engine reads.

The one fact to internalize: **geometry decides, scripts derive, `oflm add` runs.** A
model is the numbers in its weights, not its repo name or a registry entry. If those
numbers have a family recipe and a kernel set, the model runs; a new size is built by
the same command; a new architecture is the only thing that needs kernel code.

```
weights (GGUF / q4nx / safetensors)
   -> geometry     hidden, layers, heads, head_dim, rotary, vocab, FFN, layer types
   -> family       the recipe that turns those numbers into a layer program
   -> spec         the geometry as JSON (open_kernels/recipes/specs/*.json)
   -> kernel set   src/xclbins/<set>/open_kernels/ (built, never committed)
   -> link         <model>/open_kernels -> that set
   -> pack         .q4nx already packed; .npue packed on first load; GGUF streams
```

---

## The one command

```bash
# every family, every size, engine + kernels + package
cmake --workflow --preset linux-package

# one new size, end to end (SPEC -> BUILD -> GATE -> CATALOGUE -> REBUILD -> REGISTER)
python utilities/build-new-spec.py --repo Qwen/Qwen3-8B
python utilities/build-new-spec.py --repo Qwen/Qwen3-8B --dry-run

# kernels only, all specs or one
python utilities/export-kernels.py
python utilities/export-kernels.py --specs qwen3-8b
```

A user adds a model with `oflm add <repo>` (see the
[CLI reference](/docs/instructions/cli/#-add-a-model-oflm-add)). With a source
checkout, `oflm add` writes the derived spec and runs `build-new-spec.py` itself when
the size has no set yet; no command is handed to the user.

`build-new-spec.py`'s stages, and why each exists:

| stage | what runs | why |
|---|---|---|
| SPEC | `utilities/gen-kernel-spec.py` | geometry from `config.json`, or from the GGUF header when the repo has no config. A fine-tune whose base has a spec links to it. |
| BUILD | `export-kernels.py` with `OPEN_KERNELS_UNVALIDATED=1` | proves the design compiles at the new geometry. |
| GATE | `designs/attn/` + `utilities/op-gate.py` | runs the production attention worker against an fp64 oracle, and every other op through its fixture. A point enters the catalogue only after it passes. |
| CATALOGUE | `utilities/catalogue-add-point.py` | records the validated point so the next build needs no override. |
| REBUILD | `export-kernels.py` without the override | ships a build the catalogue allows. |
| REGISTER | `utilities/register-spec.py` | adds the spec hash (and name, when needed) to the frozen test lists. Mechanical; never hand-edit those hashes. |

Only GATE needs an NPU. Nothing here needs the model's weights, a container, or the
engine. `--dry-run` stops after SPEC and prints the recipe walls and the catalogue
points the size would need.

A geometry the recipe cannot build (a width that does not fit a core, `hidden` not a
multiple of the 256-wide q4 k-tile, a split FF with no twin) is moved to
`open_kernels/recipes/candidates/` with the reason. It is not thrown away.

---

## The artifacts, and what owns what

| artifact | path | owner | changes when |
|---|---|---|---|
| geometry | the weights (GGUF header / config / q4nx header) | the model | never |
| spec | `open_kernels/recipes/specs/*.json` | `gen-kernel-spec.py` / `oflm add` | a new size or a corrected geometry |
| recipe | `open_kernels/recipes/{dense,qwen35,qwen36moe,lfm2}.py` | kernel author | a new family or layout rule |
| design | `open_kernels/designs/<op>/` | kernel author | a new op or datapath |
| catalogue | `open_kernels/recipes/catalogue.py` | `op-gate.py` / `catalogue-add-point.py` | a validated hardware run |
| manifest | `src/xclbins/<set>/open_kernels/manifest.json` | the recipe, generated | every export |
| kernel set | `src/xclbins/<set>/open_kernels/` | `export-kernels.py` | every build; never committed |
| container | `model.q4nx` / `<model>.npue` | `q4nx-build` / the engine | conversion or first load |

The engine (`src/open_qwen36`) is an interpreter of `manifest.json`. No model constant
lives in C++, which is why one kernel set serves every model of a geometry.

---

## Adding a size

```bash
python utilities/build-new-spec.py --repo Qwen/Qwen3-8B
```

For a GGUF-only repo the same command works: `gen-kernel-spec.py --repo` falls back to
the GGUF header when `config.json` is 404. For a gated repo, download the GGUF once and
use the local file:

```bash
python utilities/gen-kernel-spec.py --gguf model.gguf \
    --out open_kernels/recipes/specs/<name>.json
python utilities/build-new-spec.py --spec <name>
```

A fine-tune never gets its own kernels; if the base's spec covers the geometry,
`gen-kernel-spec.py` writes a `link_to` spec and there is nothing to build.

---

## Adding a family

Do this only when the architecture's arithmetic is genuinely new. A new size of an
existing family is the command above.

1. **Derive the spec.** Add the architecture to `recipes/spec.py` (`GGUF_FAMILIES` for
   the GGUF key names, and the HF config path if it is a new `model_type`). Family
   differences that are constants of the architecture (post-RoPE norm, q/k/v bias,
   sliding-window pattern) go in the recipe, not the spec, so existing spec hashes do
   not move.
2. **Write the recipe.** Add a module under `open_kernels/recipes/`, expose the surface
   listed in `recipes/families.py`, and register it in `FAMILIES` / `family_module()`.
3. **Reuse or write the designs.** `designs/dense/dx.py` covers GQA + gated FFN;
   `designs/layer_x` covers the hybrid/MoE layer. A truly new op gets its own design.
4. **Prove the points.** Run the op's fixture (or the attention gate) and add the
   validated point to the catalogue. A point is not "supported" because the macro
   compiles; it is supported because the fixture passed.
5. **Add the spec and export.** `build-new-spec.py --repo ...` now works for every size.
6. **Record what is different** in `.agents/skills/open-kernel-families` -- not a new
   per-model skill.

If the arithmetic is not implemented, say so by name: `recipes/families.py`'s
`NOT_IMPLEMENTED` is the pattern. A model routed to the nearest recipe would emit
kernels that drop a whole stage of its layer and then "match" a replica making the same
mistake.

---

## Build environments

- **Linux (canonical):** `source utilities/env.sh` assembles XRT, ROCm/HIP, MLIR-AIE +
  Peano and the tree's paths. `cmake --workflow --preset linux-package` configures,
  builds the engine, exports every kernel spec, runs the tests and packages RPM + TGZ
  (`linux-package-deb` adds the DEB on Debian/Ubuntu).
- **Native Windows:** a downloaded XRT SDK zip + mlir-aie's own `iron_setup.py`, no WSL.
  Check the mlir-aie clone out at **v1.4.2** (see `docs/BUILD.md`).
- **Never run two exports in one checkout at once.** Each regenerates
  `designs/layer_x`'s kernel TUs for its own spec; use a `git worktree` for a parallel
  baseline.

---

## Verifying your change

```
oflm check <tag>
oflm run <tag>                 # or oflm serve <tag>; the real test
oflm-test --llm --model <tag>  # or --embedding / --vision / --tools
cmake --workflow --preset linux-package
```

The kernel export is incremental by `build_key` (recipe + every kernel source + spec +
quant). An unchanged set is skipped; `--force` rebuilds. `utilities/export-kernels.py
--check <previous-export>` compares instruction streams byte-for-byte and allows only
the build stamps in the xclbin to differ.

Before you push, run the settings check the way CI runs it (it merges the target
branch in, so a variable added elsewhere cannot break you):

```bash
utilities/check-settings-ci.sh
```

---

## Failure map

| message | what it means | what to do |
|---|---|---|
| `no open kernel set can run this model` | no installed set serves the geometry | run with a checkout so `oflm add` builds it; or install the set the release ships |
| `this build of the set predates GGUF-direct` | the set has no `gguf` manifest section | re-export with `python utilities/export-kernels.py` |
| `the recipe cannot build this shape` | a design wall | the spec is parked in `recipes/candidates/`; the shape needs a design change |
| `the catalogue has not validated N point(s)` | a gate has not run at this geometry | `build-new-spec.py` runs them; do not ship with `OPEN_KERNELS_UNVALIDATED=1` |
| `no recipe for family` / `not implemented` | the architecture has no recipe | that is the family work above; do not route it to the nearest recipe |
| `no open kernels found` at load | the model has no `open_kernels` link and no set matches | re-run `oflm add`, or check `OFLM_OPEN_KERNELS_DIR` |

---

## Branch naming

- `feat/llama-8b-kernels` -- a new size or family
- `fix/...` / `perf/...` / `docs/...` / `chore/...` as usual

---

## Getting help

- **Open an issue** before starting major kernel contributions.
- Ask on PRs or in issues.

**Kernel-specific resources:**
- `.agents/skills/open-kernel-pipeline/SKILL.md` -- the one-command path and failure map.
- `.agents/skills/open-kernel-families/SKILL.md` -- recipes, designs, geometry traps.
- `.agents/skills/open-gguf-direct/SKILL.md` -- the GGUF front door.
- `src/create_new_model.md` -- the geometry-first onboarding doc with the settings section.
- `open_kernels/README.md` -- the open engine kernel structure.
- `docs/BUILD.md` -- full build setup.

---

## License

OpenFlowLM is released under the MIT License. See
[LICENSE](https://github.com/Atomic-Germ/OpenFlowLM/blob/main/LICENSE) for details.
