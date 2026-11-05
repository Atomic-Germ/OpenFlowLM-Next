# Adding a Model to OpenFlowLM

> **Two cases, and only the second needs code.**
>
> 1. **The family has open kernels** (almost every model). There is nothing to
>    write. `oflm add <repo>` derives the geometry from the weights, links the
>    family kernel set, and — if this size has never been built — builds it
>    from the spec. A fine-tune or derivative links its base's kernels.
> 2. **The architecture is genuinely new.** Add a recipe and, if the arithmetic
>    is new, a design. That is the only path that touches C++ or kernel code.

The rule behind both: **geometry decides, scripts derive, `oflm add` runs.** A
model is not its repo name, its fine-tune name, its quant suffix, or its
`model_list.json` entry. It is the numbers in its weights. If those numbers
have a family recipe, the model runs; if the size is new, the same command
builds it.

## The short version

```bash
oflm add Qwen/Qwen3.5-4B-GGUF
oflm run qwen3.5:4b
```

```bash
oflm add ./Qwen3.8-35B-A3B-Distill-i1-GGUF      # a local directory or a .gguf
oflm add mradermacher/Some-Model-i1-GGUF:Q4_K_M # pin a quant
oflm add <repo> --dry-run                       # print the plan, change nothing
```

`oflm add` accepts a HuggingFace repo id, a ModelScope id (`--modelscope`), a
full URL, a local directory, or a local GGUF file. It installs the weights,
links the kernel set, registers the tag in `settings.json`, and the first run
packs whatever container the engine needs. No manual export, manifest edit,
hand copy, or `sudo`.

Verify the install:

```bash
oflm list
oflm check <tag>
oflm run <tag>
oflm-test --llm --model <tag>        # or --embedding / --vision / --tools
```

## What `oflm add` actually is

It is a product surface, not a single script:

- `src/src/main.cpp` dispatches `oflm add` before the normal argument parser.
- `src/pull/add_command.cpp` hands the arguments to the bundled
  `utilities/oflm-add/oflm_add` (or a separately installed `oflm-add`).
- The installer is the single source of truth for the install plan. The engine
  consumes what it wrote: `settings.json`, `<model>/open_kernels`, and the
  model directory.

What the installer does, in order:

1. **Reads the weights first.** For a GGUF, the header names the architecture,
   widths, rope law, vocabulary and tensor types. A repo's `config.json` is a
   hint for what the file does not carry (tokenizer, chat template, vision
   tower), never the geometry. For a q4nx/safetensors repo it derives the same
   geometry from the config and container header.
2. **Derives the family from the content**, then the base-model chain, then
   (last) the kernel set or the name. `--family` is an override for a model
   that names nothing; it is not required.
3. **Finds a kernel set by geometry**, not by name: the installed set whose
   `manifest.json` serves this spec and (for GGUF) carries the `gguf` section.
   The newest manifest wins, so a fresh rebuild beats a stale installed copy.
4. **Builds a missing size.** With a source checkout present, it writes the
   spec it derived and runs `utilities/build-new-spec.py --spec <name>`. No
   command is handed to the user. Without a checkout (an installed package) it
   says the size is ours to ship.
5. **Links the set** at `<model>/open_kernels` and writes the model entry to
   `settings.json`. It does not edit the shipped `model_list.json`.
6. **Packs on first load.** A `.npue` embedding container is produced locally
   from the checkpoint (or from a GGUF staged beside it) the first time the
   model runs. A GGUF chat model streams its pools directly. A `.q4nx` is
   already a container.

`model_list.json` is a shipped suggestion list for `oflm list` / `oflm pull`.
It is **not** a registry, gate, oracle, or source of truth, and it does not
even match the open kernels: a size can run with no entry, and an entry can
name a size no kernel set covers. Maintain it only as a hint of what a user
*can* add — for new families and sizes — never as the source of what works.
The registry that matters is `settings.json` (`models` section, written by
`oflm add`), merged with the shipped list for display only.

### Useful flags

| flag | meaning |
|---|---|
| `--tag NAME` / `--name NAME` | local tag (default: derived from the repo and the model's own numbers) |
| `--family NAME` | override the family when the content names nothing; not normally needed |
| `--dry-run` | print the plan and exit |
| `--text-only` | GGUF installs only: do not fetch the vision projector |
| `--mmproj REPO` | take the vision projector from another repo |
| `--open-kernels DIR` | point at a locally exported set instead of scanning |
| `--xclbin-from REF` | link a named set's xclbins |
| `--no-xclbin` | install weights without linking kernels |
| `--modelscope` | pull from ModelScope |
| `--models-root DIR` | install into another model store |
| `--force` | overwrite an existing installation |
| `--no-verify` | skip sha256 verification |
| `--quiet` / `--verbose` | less output / show passed-over GGUF candidates |

## What "supported" means

A model runs on open kernels when its **geometry** has a family recipe and a
kernel set:

| family | recipe | what it covers |
|---|---|---|
| `qwen2`, `qwen3`, `llama3`, `gemma3`, `hunyuan`, `granite`, `phi3`, `k2` | `recipes/dense.py` | GQA + gated FFN, one dispatch per layer |
| `qwen35` | `recipes/qwen35.py` | Qwen3.5/3.8 dense, 3 linear : 1 full DeltaNet |
| `qwen36moe` | `recipes/qwen36moe.py` | Qwen3.5/3.6-MoE |
| `lfm2` | `recipes/lfm2.py` | short-conv + full-attention hybrid |

The geometry is: hidden size, layer count and layer types, attention heads and
key/value heads, head dim, rotary dim and rope law, vocabulary, FFN width,
activation, norms, and quant format. `ModelSpec` holds it; the recipe turns it
into a manifest; the engine interprets the manifest. No model constant lives
in C++.

A fine-tune is not a family. If the base model's spec already covers the
geometry, `oflm add` links the base's kernels; `gen-kernel-spec.py` writes a
`link_to` spec instead of a new set.

## Adding a size

If `oflm add` says the size has no set, it builds one when a checkout is
present. To do it directly:

```bash
python utilities/build-new-spec.py --repo Qwen/Qwen3-8B
python utilities/build-new-spec.py --repo Qwen/Qwen3-8B --dry-run
```

The stages, and why each exists:

| stage | what runs | why |
|---|---|---|
| SPEC | `utilities/gen-kernel-spec.py` | geometry from `config.json`, or from the GGUF header when the repo has no config. A fine-tune whose base has a spec links to it. |
| BUILD | `export-kernels.py` with `OPEN_KERNELS_UNVALIDATED=1` | proves the design compiles at the new geometry. |
| GATE | `designs/attn/` + `utilities/op-gate.py` | runs the production attention worker against an fp64 oracle, and every other op through its fixture. A point enters the catalogue only after it passes. |
| CATALOGUE | `utilities/catalogue-add-point.py` | records the validated point so the next build needs no override. |
| REBUILD | `export-kernels.py` without the override | ships a build the catalogue allows. |
| REGISTER | `utilities/register-spec.py` | adds the spec hash (and name, when needed) to the frozen test lists. Mechanical; never hand-edit those hashes. |

Only GATE needs an NPU. Nothing here needs the model's weights, a container, or
the engine. `--dry-run` stops after SPEC and prints the recipe walls and the
catalogue points the size would need.

**GGUF-only repos** work with the same command. `gen-kernel-spec.py --repo`
falls back to the repo's GGUF header when `config.json` is 404. For a gated
repo, download the GGUF once and use the local file:

```bash
python utilities/gen-kernel-spec.py --gguf model.gguf \
    --out open_kernels/recipes/specs/<name>.json
python utilities/build-new-spec.py --spec <name>
```

**A design wall is not a failure to retry.** If the recipe cannot build the
shape (a width that does not fit a core, `hidden` not a multiple of the
256-wide q4 k-tile, a split FF with no twin), `build-new-spec.py` moves the
spec to `open_kernels/recipes/candidates/` and names the reason. A recipe that
later learns the shape picks it up.

**Rebuild and ship.** `cmake --workflow --preset linux-package` is the
canonical build: engine, every kernel spec, tests and package. For kernels only,
`python utilities/export-kernels.py` (all specs) or `--specs <name>` (one).
`export-kernels.py` is incremental by `build_key`; `--force` rebuilds.

## Adding a family

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
6. **Record what is different** in `.agents/skills/open-kernel-families` — not
   in a new per-model skill.

For a `.q4nx` container, the converter is `utilities/q4nx-build`; it must stay
in sync with the family's tensor names and quant formats. A GGUF-direct model
does not need a converter at all: the engine reads the file.

If the arithmetic is not implemented, say so by name. `recipes/families.py`'s
`NOT_IMPLEMENTED` is the pattern: a model routed to the nearest recipe would
emit kernels that drop a whole stage of its layer and then "match" a replica
making the same mistake. A named refusal is correct.

## Containers, kernels, and linking

| artifact | who produces it | where it lives |
|---|---|---|
| `.q4nx` container | `utilities/q4nx-build` | the model directory |
| GGUF | the model author | `model.gguf` in the model directory |
| `.npue` embedding container | the engine, on first load | `<model>/<name>.npue` |
| kernel set | `utilities/export-kernels.py` | `src/xclbins/<set>/open_kernels/` |
| link | `oflm add` | `<model>/open_kernels` |

Kernel sets are **built, not committed** (`src/xclbins/*/open_kernels` is
git-ignored). The package ships them pre-built. A model never carries its own
kernels; it links a family set by geometry. The engine looks for
`<model>/open_kernels` first, then the installed xclbin roots.

## Verifying

- `oflm check <tag>` — files present and compatible.
- `oflm run <tag>` / `oflm serve <tag>` — the real test.
- `oflm-test --llm --model <tag>` (or `--embedding`, `--vision`, `--tools`).
  It currently checks `finish_reason` and length; read the generated text too.
- GGUF numerics: `open_kernels/model/gguf_ref.py` +
  `replica_prompt.py --model-dir <gguf install>` runs the fp64 reference on the
  file itself.
- Embeddings: `utilities/gguf-embed-check.py`, `oflm bench-embed`.

## Legacy: closed-engine models

This path is **not** the supported way to add a model, and it is not needed for
anything with open kernels. It exists for a model that has a prebuilt closed
engine library and no open recipe.

The shape is: an engine header (`include/models/<model>/<model>_npu.hpp`), a
prebuilt `lib<model>_npu.so` / `.dll`, an `AutoModel` wrapper
(`include/AutoModel/modeling_<model>.hpp` + `common/AutoModel/modeling_<model>.cpp`),
four edits in `include/AutoModel/all_models.hpp`, the engine library in
`CMakeLists.txt`, a `model_list.json` entry (or a `settings.json` entry written
by `oflm add`), a `model_info.json` manifest for `oflm pull`, and a
`test/<model>/` harness. Copy the closest wrapper: `modeling_gemma4e.hpp`
(thought/tool markers), `modeling_qwen3.hpp` (`think` tags), `modeling_gpt_oss.hpp`
(harmony channels), `modeling_qwen3vl.hpp` (vision-language).

The old step-by-step version of this guide is in the git history before the
geometry-first rewrite. Keep new work on the open path unless you are
specifically replacing a closed engine.

---

## Adding a new setting

User-facing settings are defined in **one place**: `src/config/settings.json`.
The C++ registry the binary compiles in is **generated** from it, along with
the help text, the interactive picker, the schema and the validation. Do not
edit `src/config/settings_registry_generated.hpp` — it is overwritten.

### Step 1: Add to settings.json

Add an entry to the `settings` array:

```json
{
  "key": "decode_mode",                    // Canonical name in settings.json
  "cli_flag": "--decode-mode",             // CLI flag (empty if none)
  "env_var": "OFLM_OPEN_DECODE_MODE",      // Env var (for ENV_VAR delivery)
  "description": "Decode mode: exact or fast",
  "type": "enum",
  "options": [
    {"value": "exact", "description": "Bit-exact, slower"},
    {"value": "fast", "description": "Faster, approximate"}
  ],
  "min_value": null,
  "max_value": null,
  "presets": [],
  "default_value": "exact",
  "scope": "per_model",                    // global, per_model, or both
  "reload_required": true,                 // Needs model reload if changed
  "category": "performance",                // Grouping for help/picker
  "registry_key": "",                      // Maps to model_list.json entry key
  "delivery": "env_var"                    // How the value reaches the engine
}
```

### Step 2: Choose the delivery mechanism

The `delivery` field controls how the setting's value reaches the code that
consumes it:

| Delivery | When to use | What happens |
|----------|-------------|--------------|
| `env_var` | Value must cross the `OFLM_DLL` boundary | `apply_settings()` calls `setenv()` automatically |
| `direct` | Same process, passed via function parameter | Value is in the resolved JSON; consumer reads it |
| `api_field` | Per-request override via REST API | Value is used as default; API field overrides |
| `registry` | Value lives in the model registry entry | Read from merged registry at load time |

**For engine options that cross the DLL boundary, use `env_var`.** The engine
reads the env var (one line: `getenv_oflm("OFLM_OPEN_DECODE_MODE")`), and
`apply_settings()` setenvs it automatically. No glue code needed.

The value the engine reads must match what the settings system delivers: a
boolean arrives as `true`/`false`, so a switch the engine tests with `== "1"`
has to accept both.

### Step 3: Engine reads the value

```cpp
std::string decode_mode = utils::getenv_oflm("OFLM_OPEN_DECODE_MODE");
if (decode_mode.empty()) decode_mode = "exact"; // default
```

### Step 4: Classify every flag and variable the code reads

CI scans the source tree for CLI flags and `OFLM_*` variables, and fails if
one is neither a setting nor explicitly excluded. If the code reads
`OFLM_SOMETHING` and it is **not** a user-facing setting, list it in
`NON_SETTING_ENV_VARS` in `utilities/generate-registry.py`, with a reason.

### Step 5: Regenerate, build and test

```bash
cmake --build --preset linux-default --target generate-settings
git diff --exit-code src/config/settings.json src/config/settings_registry_generated.hpp

cmake --build --preset linux-default --target oflm
./build/src/oflm config gen-schema | python3 -m json.tool | grep decode_mode
./build/src/oflm config show
ctest --preset linux-default
```

The new setting automatically appears in:
- `oflm config --help`
- `oflm config show`
- `oflm config set` (with validation)
- Interactive picker
- Shipped `model_list.json` schema (at package time)

**Before you push, run the check the way CI runs it** — it merges the target
branch in, so a variable the target branch added does not break you:

```bash
utilities/check-settings-ci.sh
```

### Field reference

| Field | Required | Description |
|-------|----------|-------------|
| `key` | Yes | Canonical name in settings.json and API |
| `cli_flag` | No | CLI flag (e.g. `--decode-mode`), empty if none |
| `env_var` | If env_var delivery | Environment variable name |
| `description` | Yes | Human-readable description |
| `type` | Yes | `enum`, `int`, `bool`, or `string` |
| `options` | If enum | Valid options with descriptions |
| `min_value`/`max_value` | If int | Value range |
| `presets` | No | Quick-pick values for int in picker |
| `default_value` | No | Fallback default (empty = from registry entry) |
| `scope` | Yes | `global`, `per_model`, or `both` |
| `reload_required` | Yes | `true` if changing requires model reload |
| `category` | Yes | Grouping: `performance`, `memory`, `generation` |
| `registry_key` | No | Maps to model_list.json entry key (e.g. `default_context_length`) |
| `delivery` | Yes | `direct`, `env_var`, `api_field`, or `registry` |

---

## Checklist

**A model whose family has kernels**

- [ ] `oflm add <repo>` succeeds (use `--dry-run` first if unsure).
- [ ] `oflm check <tag>` passes.
- [ ] `oflm run <tag>` produces coherent text (or `oflm-test --embedding` for an embedding model).
- [ ] `<model>/open_kernels` points at a set whose manifest serves the geometry.
- [ ] The tag appears in `oflm list`.

**A new size**

- [ ] `python utilities/build-new-spec.py --repo <repo>` completes (or `--dry-run` shows only points the gates can cover).
- [ ] The spec is in `open_kernels/recipes/specs/`, not `candidates/`.
- [ ] `utilities/register-spec.py` touched the frozen test lists; commit those files with the spec.
- [ ] `cmake --workflow --preset linux-package` still passes.

**A new family**

- [ ] `recipes/spec.py` derives the geometry from both HF config and GGUF metadata.
- [ ] The recipe module exposes the `families.py` surface and is registered.
- [ ] Every catalogue point the family needs has passed its fixture.
- [ ] A spec builds and the engine loads it; `open-kernel-families` records the traps.
