---
name: open-kernel-pipeline
description: The one-command path from weights to a running model (geometry -> family -> spec -> kernel set -> link -> pack). Use when adding a model or size, onboarding a repo with `oflm dev build-new-family`, when `oflm add` refuses a geometry, when deciding whether a change is a spec, recipe, or design change, or when rebuilding and shipping kernel sets. Supersedes the per-family manual export workflows.
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

## Three front doors, one pipeline

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

**Proven end-to-end from a public GGUF repo, with no hand-written spec:**

```bash
oflm dev build-new-spec --repo mradermacher/Yi-1.5-9B-Chat-i1-GGUF
cmake --workflow --preset linux-package
sudo dnf reinstall -y build/packages/openflowlm-0.1.0-1.x86_64.rpm
oflm add mradermacher/Yi-1.5-9B-Chat-i1-GGUF
oflm run yi-chat:9b        # answers correctly on the NPU
```

That is the whole onboarding for a shape-identical model: derive spec, build,
gate, catalogue, register, package, install, run. The only case that still
needs a design change is a size whose widths do not divide into whole 64-row
bands / 256-wide k-tiles (a geometry wall, not a tooling gap).

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

### For an onboarding run: `oflm dev build-new-family <repo>`

```bash
oflm dev build-new-family Qwen/Qwen3-0.6B --dry-run
oflm dev build-new-family mradermacher/Qwen2.5-7B-Instruct-i1-GGUF
oflm dev build-new-family unsloth/Qwen3.8-Flash-Next-GGUF --dry-run
```

`utilities/build-new-family.py` is an orchestrator in front of the two commands
above, not a second pipeline. It still lets `gen-kernel-spec.py` decide geometry
and `build-new-spec.py` own SPEC -> BUILD -> GATE -> CATALOGUE -> REBUILD ->
REGISTER. What it adds is the front door for "onboard this repo".

**It is not a sibling of `build-new-spec`.** `build-new-spec` takes a known
recipe and *finishes* a size end to end. `build-new-family` **begins a port
effort**: it tries building the new model against the nearest existing recipe,
and if that succeeds the case was only a new size / new spec after all (Bonsai
needed just a Q2_0 GGUF reader, not a family). If it hits a wall, that wall is
the signal that real arithmetic port work is required — the tool hands you the
evidence and stops. Do not read a `would_extend` result as "the family was
built".

1. **SPEC** — GGUF-header first; for a GGUF-only repo it scans **every** GGUF
   path, so a dynamic quant in `UD-Q4_K_XL/...-00001-of-00004.gguf` is a geometry
   source. The repo README's `base_model` chain is recorded as the tensors/config
   authority only when config/tokenizer is needed; a local `--gguf` works for
   gated repos. No q4nx builder is involved (q4nx is deprecated).
2. **CLASSIFY** — `served` (a checked-in spec/link covers this kernel geometry
   **and** the install probe found a loadable GGUF), `would_build` (known family,
   new size), `would_extend` (an architecture no recipe owns), or `weights_blocked`
   (the geometry hash matched, but the installer refused every GGUF — a ternary
   quant with unknown ggml tensor types is not an installable model). The
   `oflm add --dry-run` probe reads `gguf kernels` for GGUF
   repos and `xclbin source` for config-based repos; its closest official match
   is **not** used as geometry evidence. The probe is skipped for `would_extend`:
   there are no kernels to link until the extension builds.
3. **BUILD / INSTALL** — for `would_build` and `would_extend` it promotes the
   derived spec and runs `build-new-spec.py`, then `oflm add` links/registers the
   model. `served` skips the build and only needs `oflm add`.
4. **UNKNOWN ARCHITECTURE** — no recipe owns the architecture. The command reads
   the model's own GGUF header and tensor table, builds a candidate `ModelSpec`
   from the standard GGUF keys, scores the checked-in recipes against that
   geometry by **recipe capability** (expert / gate / conv — from the recipe
   family, never from a possibly mislabelled spec file), and tries the nearest
   one. `--family` picks a different recipe. `build/new-family/<repo>/RESEARCH.md`
   records which recipe was chosen, the unmatched tensors/metadata keys and the
   references, so the numeric gates can be trusted instead of a successful
   export. If the recipe cannot build the shape, `build-new-spec` parks the spec
   in `recipes/candidates/` and the command reports that path — that is the port
   boundary, not a failed install. A family can still turn out to need only a
   reader change (Bonsai/Q2_0) rather than a new recipe; start from the evidence
   before deciding it needs a port. Auxiliary GGUFs (`mmproj-*`, `dspark-*`,
   `draft-*`, `mtp-*`) are excluded from model selection, so a draft head's
   `dflash` architecture cannot name the family.
5. **VALIDATE** — layer dumps, `cmp_layers.py`, decode gates and the production
   bar are printed as required next steps, and run here only when `--cli`,
   `--ref-model` and `--install-dir` are supplied. A successful build is not a
   support claim.

`--dry-run` writes only temporary artifacts (candidate spec, build report, probe
log, JSON plan); it does not promote a spec, build kernels or edit settings.
Useful overrides are `--family` (the recipe to extend; also a conservative `oflm
add` hint), `--tensor-repo`, `--spec-name`, `--gguf`, and the `--skip-build` /
`--skip-install` / `--skip-validate` stage stops. A quant can be named as
`user/repo:QUANT`; the bare repo is used for spec derivation and the quant for
install/GGUF selection.

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

### Unsloth UD quants and split files

`oflm add` accepts `repo:UD-Q4_K_XL` (and `:UD-IQ4_XS`). The `UD-` prefix is a
dynamic mix, not a tensor type: strip it and verify the real ggml types in the
header. `UD-Q4_K_XL` mixes Q4_K gates/ups with Q5_1 MoE downs, Q8_0 projections,
F32 norms and BF16 indexer tensors; the packers now ingest Q5_0/Q5_1 as well.

Big Unsloth exports are split. Install part 1 as `model.gguf` and its siblings
as `model.gguf-00002-of-00004` (no trailing `.gguf` on the siblings). `GgufFile`
maps every shard, merges the tensor tables and remembers each tensor's shard;
`read_gguf_inventory()` does the same on header prefixes, with shard 1 as the
metadata authority. Unsloth's `split.no` is **0-based** (llama.cpp's is
1-based), so infer the convention from the first shard and trust the
`-NNNNN-of-NNNNN` file names for order. A set whose metadata lives in part 1
and tensors in the later parts reads as zero tensors if only part 1 is parsed.
These are weight-layout facts, not a new family: do not add kernels for a
GGUF's `general.architecture` unless its arithmetic is actually new.

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
| `allocated buffers exceeded available memory` (aiecc, with a `MemoryMap`) | a fixture's fifo buffers + stack exceed the AIE tile's 64 KB at this width (e.g. `ln` at N=8192 uses four 16 KB elements) | a DESIGN wall in that op's streaming scheme, not a spec error; `build-new-spec` translates the raw aiecc message. Needs a narrower/split path in the design, like `ln.py`'s `SPLIT` branch |
| `the catalogue has not validated N point(s)` | a gate has not run at this geometry | `build-new-spec.py` runs them; do not ship with `OPEN_KERNELS_UNVALIDATED=1` |
| `no recipe for family` / `not implemented` | the architecture has no recipe | that is the family work above; do not route it to the nearest recipe |
| `status: would_extend` from `build-new-family` | the GGUF/config names an architecture no recipe owns | the nearest recipe was tried as a starting point; a clean build means it was only a new size, a wall means begin the port with `open-family-addition`. Read `build/new-family/<repo>/RESEARCH.md` either way |
| `status: weights_blocked` from `build-new-family` | the geometry hash matched a checked-in spec, but `oflm add --dry-run` refused every GGUF (unsupported quant or unknown tensor type) | no build/install/gate is attempted; the weights must be converted to a supported quant first -- a geometry match does not make a weight file loadable |
| `no open kernels found` at load | the model directory has no `open_kernels` link and no set matches | re-run `oflm add`, or check `OFLM_OPEN_KERNELS_DIR` |
| `model.gguf` present but no `gguf` section | a GGUF install linked a q4nx-only set | re-export the set; the engine swaps the whole manifest view |

## Where to go next

- `open-kernel-families` — what each recipe/design does, and the geometry traps.
- `open-gguf-direct` — the GGUF front door, pool layout, quant support, packing.
- `open-diffusion` / `open-diffusion-edits` — FLUX.2 klein, a separate schedule.
- `open-dxl-rows` — the dense L-row pass behind speculative decoding.
- `open-norm-bf16-validation` — RMSNorm BF16 output paths.
- `openflowlm-packaging` / `openflowlm-release` — shipping the built sets.
