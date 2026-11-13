---
name: open-kernel-pipeline
description: The one-command path from weights to a running model (geometry -> family -> spec -> kernel set -> link -> pack). Use when adding a model or size, when `oflm add` refuses a geometry, when deciding whether a change is a spec, recipe, or design change, or when rebuilding and shipping kernel sets. Supersedes the per-family manual export workflows.
---

# The open-kernel pipeline

**One rule: geometry decides, scripts derive, `oflm add` runs.** A model is not
its repo name, its fine-tune name, its quant suffix, or its `model_list.json`
entry. It is the numbers in its weights. If those numbers have a family recipe
and a kernel set, the model runs. If the size is new, the same command builds
it. If the architecture is new, that is the only case that needs kernel code.

```
weights (GGUF / q4nx / safetensors)
   -> geometry        hidden, layers, heads, head_dim, rotary, vocab, FFN, layer types
   -> family          the recipe that turns those numbers into a layer program
   -> spec            the geometry as JSON (open_kernels/recipes/specs/*.json)
   -> kernel set      src/xclbins/<set>/open_kernels/ (built, never committed)
   -> link            <model>/open_kernels -> that set
   -> pack            .q4nx already packed; .npue packed on first load; GGUF streams
   -> serve           oflm run / serve / bench
```

## Two front doors, one pipeline

### For a user: `oflm add`

```bash
oflm add <hf-repo | modelscope-repo | url | local-dir | local.gguf>
oflm add <...> --dry-run          # print the plan, change nothing
```

`oflm add` is a product surface, not just a Python utility:

- `src/src/main.cpp` dispatches `oflm add` before the normal argument parser.
- `src/pull/add_command.cpp` hands the arguments to the bundled
  `utilities/oflm-add/oflm_add` (or a separately installed `oflm-add`).
- The installer is the single source of truth for the install plan; the engine
  consumes what it wrote.

What it does, in order:

1. **Reads the weights first.** For a GGUF, the header names the architecture,
   widths, rope law, vocabulary and tensor types. A repo's `config.json` is a
   hint for what the file does not carry (tokenizer, chat template, vision
   tower), never the geometry. For a q4nx/safetensors repo it derives the same
   geometry from the config and container header.
2. **Derives the family from the content**, then the base-model chain, then
   (last) the kernel set or the name. `--family` is an override for the rare
   model that names nothing; it is not required.
3. **Finds a kernel set by geometry**, not by name: the installed set whose
   `manifest.json` serves this spec and (for GGUF) carries the `gguf` section.
   Newest manifest wins, so a fresh rebuild beats a stale installed copy.
4. **Builds a missing size.** With a source checkout present, it writes the
   spec it just derived and runs `utilities/build-new-spec.py --spec <name>`.
   No command is handed to the user. Without a checkout (an installed package)
   it says the size is ours to ship.
5. **Links the set** at `<model>/open_kernels` and writes the model entry to
   `settings.json`. It does not edit the shipped `model_list.json`.
6. **Packs on first load.** A `.npue` embedding container is produced locally
   from the checkpoint (or from a GGUF staged beside it) the first time the
   model runs. A GGUF chat model streams its pools directly. A `.q4nx` is
   already a container.

`model_list.json` is a shipped suggestion list for `oflm list` / `oflm pull`.
It is **not** a registry, gate, oracle, or source of truth, and it does not
even enumerate what the open kernels serve: a size can run with no entry, and
an entry can advertise a size no kernel set covers. Maintain it only as a hint
for *new families and sizes* — "you can add these" — never as the set of what
works. The registry that matters is `settings.json` (`models` section, written
by `oflm add`) merged with the shipped list for display only.

### For a maintainer: one command

```bash
# every family, every size, engine + kernels + package
cmake --workflow --preset linux-package

# one new size, end to end (SPEC -> BUILD -> GATE -> CATALOGUE -> REBUILD -> REGISTER)
python utilities/build-new-spec.py --repo <hf-repo>
python utilities/build-new-spec.py --repo <hf-repo> --dry-run

# kernels only, all specs or one
python utilities/export-kernels.py
python utilities/export-kernels.py --specs qwen3-8b
```

`build-new-spec.py` is the scripted form of the advice the docs used to give.
Its stages, and why each exists:

| stage | what runs | why |
|---|---|---|
| SPEC | `utilities/gen-kernel-spec.py` | geometry from `config.json`, or from the GGUF header when the repo has no config (a quantized upload names its own geometry). A fine-tune whose base has a spec links to it instead. |
| BUILD | `export-kernels.py` with `OPEN_KERNELS_UNVALIDATED=1` | proves the design compiles at the new geometry. |
| GATE | `designs/attn/` + `utilities/op-gate.py` | runs the production attention worker against an fp64 oracle, and every other op through its fixture. A point enters the catalogue only after it passes. |
| CATALOGUE | `utilities/catalogue-add-point.py` | records the validated point so the next build needs no override. |
| REBUILD | `export-kernels.py` without the override | ships a build the catalogue allows. |
| REGISTER | `utilities/register-spec.py` | adds the spec hash (and name, when needed) to the frozen test lists. Mechanical; never hand-edit those hashes. |

Only GATE needs an NPU. Nothing in the pipeline needs the model's weights, a
container, or the engine. `--dry-run` stops after SPEC and prints the recipe
walls and the catalogue points the size would need.

A geometry the recipe cannot build (a width that does not fit a core, a
`hidden` that is not a multiple of the 256-wide q4 k-tile, a split FF with no
twin) is moved to `open_kernels/recipes/candidates/` with the reason. It is
not thrown away: a recipe that learns the shape picks it up.

## The artifacts, and what owns what

| artifact | path | owner | changes when |
|---|---|---|---|
| geometry | weights (GGUF header / config / q4nx header) | the model | never |
| spec | `open_kernels/recipes/specs/*.json` | `gen-kernel-spec.py` / `oflm add` | a new size or a corrected geometry |
| recipe | `open_kernels/recipes/{dense,qwen35,qwen36moe,lfm2}.py` | kernel author | a new family or a new layout rule |
| design | `open_kernels/designs/<op>/` | kernel author | a new op or a new datapath |
| catalogue | `open_kernels/recipes/catalogue.py` | `op-gate.py` / `catalogue-add-point.py` | a validated hardware run |
| manifest | `src/xclbins/<set>/open_kernels/manifest.json` | the recipe, generated | every export |
| kernel set | `src/xclbins/<set>/open_kernels/` | `export-kernels.py` | every build; never committed |
| container | `model.q4nx` / `<model>.npue` | `q4nx-build` / the engine | conversion or first load |

The engine (`src/open_qwen36`) is an interpreter of `manifest.json`. No model
constant lives in C++. That is why one kernel set serves every model of a
geometry, and why a new size is a spec, not a code change.

## Adding a size: the only command

```bash
python utilities/build-new-spec.py --repo Qwen/Qwen3-8B
```

For a GGUF-only repo, the same command works: `gen-kernel-spec.py --repo`
falls back to the repo's GGUF header when `config.json` is 404. For a gated
repo, download the GGUF once and use the local file:

```bash
python utilities/gen-kernel-spec.py --gguf model.gguf --out open_kernels/recipes/specs/<name>.json
python utilities/build-new-spec.py --spec <name>
```

A fine-tune never gets its own kernels. If the base model's spec already
covers the geometry, `gen-kernel-spec.py` writes a `link_to` spec and
`build-new-spec.py` stops with "the base model already has kernels".

## Adding a family: the only code work

Do this only when the architecture's arithmetic is genuinely new. A new size of
an existing family is the command above.

1. **Derive the spec.** Add the architecture to `recipes/spec.py`
   (`GGUF_FAMILIES` for the GGUF key names, and the HF config path if it is a
   new `model_type`). Put family differences that are constants of the
   architecture (post-RoPE norm, q/k/v bias, sliding-window pattern) in the
   recipe, not in the spec, so existing spec hashes do not move.
2. **Write the recipe.** Add a module under `open_kernels/recipes/`, expose the
   surface listed in `recipes/families.py` (`recipe`, `layout`, `pack_plan`,
   `programs`, `builds`, `hf_config_check`, `manifest_layout`,
   `KERNEL_SOURCES`, `GEN_KERNELS`), and register it in `FAMILIES` and
   `family_module()`.
3. **Reuse or write the designs.** `designs/dense/dx.py` covers GQA + gated FFN.
   `designs/layer_x` covers the hybrid/MoE layer. A truly new op gets its own
   design under `designs/`.
4. **Prove the points.** Run the op's fixture (or the attention gate) and add
   the validated point to the catalogue. A point is not "supported" because the
   macro compiles; it is supported because the fixture passed.
5. **Add the spec and export.** `build-new-spec.py --repo ...` now works for
   every size of the family.
6. **Record what is different** in `open-kernel-families`, not in a new
   per-model skill.

## Build environments

- **Linux (canonical):** `source utilities/env.sh` assembles XRT, ROCm/HIP,
  MLIR-AIE + Peano and the tree's paths. `cmake --workflow --preset
  linux-package` configures, builds the engine, exports every kernel spec, runs
  the tests and packages the RPM + TGZ. `linux-package-deb` adds the DEB (build
  it on Debian/Ubuntu; a DEB built on Fedora carries Fedora ABI and
  `shlibdeps` cannot resolve there). `cmake --build build --target
  clean-kernels` sweeps the export scratch.
- **Native Windows:** works, with a downloaded XRT SDK zip and mlir-aie's own
  `iron_setup.py` — no WSL, no source build. Check the mlir-aie clone out at
  **v1.4.2** before running `iron_setup.py`; a `main` checkout installs the
  rolling wheel whose IRON API these designs do not import against. The XRT SDK
  supplies `xclbinutil.exe` and `aiebu-asm.exe`; dot-source `iron_env.ps1` so
  they and `aiecc` reach the build. See `docs/BUILD.md` ("2. The NPU kernels")
  and mlir-aie's `docs/buildHostWinNative.md`.
- **WSL fallback:** Ubuntu 24.04's own XRT is 2022 (pre-NPU) and building
  `aiebu` needs root, so use an XRT build's `xclbinutil` / `aiebu-asm` under
  `~/xrt-tools`. The designs are OS-neutral; an xclbin is a device artifact.
- **Never run two exports in one checkout at once.** Each regenerates
  `designs/layer_x`'s kernel TUs for its own spec. Use a `git worktree` for a
  parallel baseline.
- **Standalone test binaries** (`open_kernels/harness`, `src/open_qwen36`)
  need `XRT_INCLUDE_DIR`, `XRT_LIB_DIR` and `VCVARS64` on Windows; the scripts
  default to VS 2022 BuildTools, so set `VCVARS64` if your VS is Community.

## Rebuild and ship

`export-kernels.py` is incremental: `recipes/cache.py` hashes the recipe, every
kernel source the designs include, the spec and the quant format. An unchanged
set is skipped; `--force` rebuilds. `open_kernels/export_qwen36_kernels.py
--check <previous-export>` compares instruction streams byte-for-byte and
allows only the build stamps in the xclbin to differ.

## Failure map

| message | what it means | what to do |
|---|---|---|
| `no open kernel set can run this model` | no installed set serves the geometry | run with a checkout so `oflm add` builds it; or install the set the release ships |
| `this build of the set predates GGUF-direct` | the set has the geometry but no `gguf` manifest section | re-export with `python utilities/export-kernels.py` |
| `the recipe cannot build this shape` | a design wall (width, k-tile, split FF, head count) | the spec is parked in `recipes/candidates/`; the shape needs a design change |
| `the catalogue has not validated N point(s)` | a gate has not run at this geometry | `build-new-spec.py` runs them; do not ship with `OPEN_KERNELS_UNVALIDATED=1` |
| `no recipe for family` / `not implemented` | the architecture has no recipe | that is the family work above; do not route it to the nearest recipe |
| `no open kernels found` at load | the model directory has no `open_kernels` link and no set matches | re-run `oflm add`, or check `OFLM_OPEN_KERNELS_DIR` |
| `model.gguf` present but no `gguf` section | a GGUF install linked a q4nx-only set | re-export the set; the engine swaps the whole manifest view |

## Where to go next

- `open-kernel-families` — what each recipe/design does, and the geometry traps.
- `open-gguf-direct` — the GGUF front door, pool layout, quant support, packing.
- `open-diffusion` / `open-diffusion-edits` — FLUX.2 klein, a separate schedule.
- `open-dxl-rows` — the dense L-row pass behind speculative decoding.
- `open-norm-bf16-validation` — RMSNorm BF16 output paths.
- `openflowlm-packaging` / `openflowlm-release` — shipping the built sets.
