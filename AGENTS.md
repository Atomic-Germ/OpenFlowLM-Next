# OpenFlowLM — working notes for agents and humans

**This is Atomic-Germ's project, not a downstream fork.** It grew out of
ROCm/FastFlowLM issues [#219](https://github.com/ROCm/FastFlowLM/issues/219),
[#246](https://github.com/ROCm/FastFlowLM/issues/246) and
[#690](https://github.com/ROCm/FastFlowLM/issues/690), which asked for closed NPU
kernels to be opened. We do not maintain closed kernels. Any that remain are
legacy, carried until an open set replaces them. Do not write new work against a
closed kernel, and do not treat one as the reference for "how this project works."

## The one thing to understand before you touch anything

**The point of this project is that a fine-tune nobody has ever heard of runs with
zero new code.** Someone publishes `Qwen3.5-9B-Instruct-NPU2-v3-my-tune`. It has
the same *shape* as a model we already support. It must therefore run, today, by
converting it and installing it — not by adding its name somewhere, and not by
writing new kernels for it.

**Three tools are the founding pieces of this project, and everything else bends to
them.** They are not thin wrappers around a downloader and a converter. Read the
source before you describe any of them, because the one-line summaries that
circulate ("it converts", "it installs") are wrong in the ways that matter:

> **Call them through `oflm`, not by their Python names.** The C++ core already owns
> both entry points:
>
> | Command | Hands off to | Lives in |
> |---|---|---|
> | `oflm pack` | `q4nx-build` | `src/pull/pack_command.cpp` |
> | `oflm add` | `oflm-add` | `src/pull/add_command.cpp` |
>
> These are **deliberate process handoffs, not stubs**, and the reasoning is written
> down in `src/pull/pack_command.hpp:9-16`. Both find the bundled tool (installed
> beside `oflm`, or in `share/oflm/utilities/`), set up the environment, and exec it
> with arguments passed through **untouched** — so the Python beside `oflm` stays the
> single source of truth for what every flag means. Overridable with
> `OFLM_PACK_SCRIPT` / `OFLM_ADD_SCRIPT` and `OFLM_PYTHON`.
>
> `q4nx-build` / `oflm-add` are the *implementation*. `oflm pack` / `oflm add` are
> the *interface*. Never add a parallel copy in either language.

### `utilities/q4nx-build` — makes any model buildable

This is the tool that decides **what a model is** and turns that into something the
NPU can run. It is the largest and most load-bearing piece here.

- **`arch_detect.py`** reads a GGUF's `general.architecture` and, when that is
  absent or wrong, fingerprints the *actual tensor names* against a per-family
  profile (keywords, distinctive tensors, required-any, excludes, metadata field
  prefixes) with explicit confidence tiers. `render_chart()` renders the whole
  chart so the heuristics can be audited in one place. A mislabeled or unlisted
  GGUF still lands on a plausible family the user can confirm or override with `-f`.
- **`build_plan.py`** walks the model card's `base_model` chain upward and stops at
  the first ancestor with an `{org}/{base}-NPU2` skeleton mirror (orgs
  `Atomic-Germ`, then `OpenFlowLM`). That skeleton supplies the config, tokenizer
  and vision assets a build needs. The chain also yields the display name, the
  parameter-size token, and whether the pipeline tag means this is a VLM. Every
  network touch is optional — offline failures degrade to a warning.
- **`models/*.py`** — 16 per-architecture converters, one per family.
- **`deploy.py`** writes the registry entry; **`--deploy-from`** copies defaults
  from an official tag.
- **`open_causal.py`, `open_embedding.py`, `open_whisper.py`, `open_diffusion.py`**
  are whole-repo builders for the *other* engines — unquantized weights packed for
  a specific kernel set, each recording per-tensor dtype and emitting
  `model_info_entry.json`. These are not the chat-model path.
- **`imatrix_prune.py`** narrows an over-wide FFN or expert set by calibration
  importance, refusing to chop blind.
- **`--build-spec`** derives the `ModelSpec` the way `oflm-add` will see it.

`q4nx-build` is where "any model works" is actually won. If a model cannot be
built, that is a gap in **this** tool before it is a gap anywhere else.

> **Why this is Python and must stay so.** `src/pull/pack_command.hpp:9-16` records
> the decision: the builder is a *quantizer*. It packs GGUF and HF-safetensors into
> a Q4NX container whose bytes must match what the NPU kernels read, and it does
> that with torch, numpy, `gguf` and `safetensors`. **Reimplementing that bit layout
> in C++ fails silently — wrong weights, not a crash** — which is the same failure
> mode that already bit `Q4_0_4_4` (installed as `Q4_0` pools). A port here buys
> nothing and risks the one failure you cannot detect.

### `utilities/oflm-add` — resolves, verifies and binds an unknown model

`oflm pull` downloads a tag that is *already in the registry*. `oflm-add` does not
require the model to be known to anyone — which is precisely why it is a separate
tool. It is a resolver and a verifier, not an installer:

- **Chooses the path.** Scans the repo for a `model.q4nx`; if none, it looks for a
  llama.cpp-style `.gguf` and installs it as `model.gguf` (the GGUF-direct path, no
  conversion needed).
- **Verifies the GGUF against its own bytes.** `parse_gguf_inventory` /
  `verify_gguf_choice` read the file's actual tensors and compare them to what the
  filename claims. `Q4_0_4_4` is a different tensor layout; matching on a prefix
  once installed it as `Q4_0` pools and produced **silent garbage**. It refuses,
  naming the quant.
- **Derives everything else.** Walks the base-model chain for skeleton assets,
  normalizes `tokenizer_config`, derives the tag, the family, and the size — guessing
  the size from GGUF bytes and quant, or from the peeked config's geometry, rather
  than refusing when a chaotic slug carries no size marker.
- **Binds kernels by shape.** Derives the `ModelSpec` from the *installed* files and
  links a matching kernel set (see below).
- **Writes a user-level registry** at `~/.config/oflm/model_list.json` and one
  symlink into `~/.config/oflm/xclbins/`. It never rewrites the system list or the
  system xclbins.

### `utilities/oflm-test` — the acceptance gate

`--llm`, `--vision`, `--embed`, `--tools` (plus server conformance). Bundled test
media, automated pass/fail verdicts, CSV results, non-zero exit on failure. **Run
it before claiming anything works.** It is the only thing standing between "the
docs say it works" and "it works".

**When a doc, a skill, a comment, or your own prior belief disagrees with these
three tools, the tools are right and the doc is wrong.** That is the tie-breaker
for this entire repository. Fix the doc.

The everyday path through the first two, which the rest of this document describes:

```bash
oflm pack -i Someone/Qwen3.5-9B-Instruct-NPU2-v3-my-tune   # build (GGUF or HF repo id)
oflm add  Someone/Qwen3.5-9B-Instruct-NPU2-v3-my-tune      # resolve, verify, bind
```

Both are the `oflm` binary; each hands off to the bundled Python beside it.

### Where this is going: the two tools merge

The GGUF-direct work exists to collapse those two commands into **one**.
`oflm-add` already installs a llama.cpp-style `.gguf` as `model.gguf` with **no
q4nx conversion at all**, and the open kernel sets ship an f32-scale build for it
(`open_kernels/gguf_pool.py`, the manifest's `gguf` section). So `q4nx-build`'s
conversion step is increasingly a *fallback* for quants the NPU pools cannot
ingest, and the intended end state is a single step:

```bash
oflm-add Someone/Some-Model-GGUF     # verify, bind, register
```

Two real constraints on that merge:

- `GGUF_CAPABLE_FAMILIES` is `{qwen3, llama3, gemma3, granite, hunyuan}`
  (`oflm_add/__init__.py:75`). qwen3.5 (fused `attn_qkv`/MTP) and qwen3.6-moe are
  not in it; a GGUF for those installs as safetensors instead. Widening this is
  kernel work, not plumbing.
- Accepted quants are `Q4_1, Q4_0, Q4_K, Q6_K, Q8_0` — exact pools for the first
  two, host requant for the K-quants. Anything else is refused and points at
  `q4nx-build`.

**Do not describe these tools as "a converter" and "an installer".** They are a
model resolver (`q4nx-build`), a model binder (`oflm-add`), and an acceptance gate
(`oflm-test`), and the GGUF work is folding the first two together.

## What actually happens when a model is installed

`oflm-add` resolves kernels through **two independent paths**. Understanding that
they are separate is most of what makes the rest make sense.

**1. Closed/legacy xclbins** — `resolve_official()` in
`utilities/oflm-add/oflm_add/__init__.py:488`. This is the upstream mechanism,
and it is **name-based**: it scores official registry entries by longest common
token prefix (`match_official_entry`, needs ≥2 tokens), then falls back to
family+size, then to the single entry for a family, then to nearest size. It exists
because custom models never shipped xclbins. **This path is the antithesis of what
this project is for.** It is why a custom fine-tune with an unusual name can get
the wrong kernels, or none. Treat it as legacy, not as the design.

**2. Open kernel sets** — `setup_open_kernels()` → `find_open_kernels()` at
`utilities/oflm-add/oflm_add/__init__.py:1356`. This is the path that matters. It is
**shape-based, not name-based**: it derives the model's `ModelSpec` from the files
it actually shipped (`recipes.load.spec_from_model_dir` — reads `config.json`, the
tokenizer's vocab, and the per-role weight format from the `model.q4nx` safetensors
header, touching no weight bytes), takes `spec.kernel_hash()`, and links any
installed kernel set whose `manifest.json` carries the same hash. **The model name
is never consulted.** A fine-tune nobody named reaches the kernels its shape calls
for, because matching is by geometry.

The kernel sets themselves are **family** sets, not per-model. One built
`Qwen3.5-9B-NPU2` serves every 9B of that shape. Family xclbins ship with the
application; the converted model ships beside them.

### `real_vocab` is NOT in the match key (fixed — do not regress this)

`spec_hash()` hashes **every** spec field, including `real_vocab` — the tokenizer's
true id count. `real_vocab` sizes nothing (`lm_rows` in `recipes/dense.py` rounds
`vocab` to a band) but it *was* in the hash, so a fine-tune that was identical to a
shipped model in every kernel-relevant respect, differing only in what its
`tokenizer.json` reported, failed to match and silently fell back to closed
kernels.

That is fixed. There are now **two** hashes, answering different questions:

- **`spec_hash`** — identity. This model, these exact hyperparameters.
- **`kernel_hash`** (`spec.py`, `spec_hash` minus `real_vocab`) — "will kernels
  built from this spec fit this model". **This is what `oflm-add` binds on.**

`manifest.json` carries both (`recipes/manifest.py`). `find_open_kernels` matches on
`kernel_hash`; a manifest **predating** the key carries only `spec_hash`, so it is
matched on identity and binds only a byte-identical spec. `oflm-add` detects that
case and says so ("*N* installed open kernel set(s) predate the shape key… their
kernels ARE among them -- re-export"), rather than telling a user to build kernels
that are already on disk. Verified against the shipped `qwen3-4b.json`:

```
              spec_hash            kernel_hash
shipped       sha256:602fa1836b218  sha256:dad4fdeedbb2
fine-tune     sha256:b3934ee0d9d90  sha256:dad4fdeedbb2
spec_hash matches?   False   <- the old bug
kernel_hash matches? True    <- now binds
```

**When you hit a non-match, do not add a whitelist and do not add the model's name
to a table.** If `kernel_hash` does not match, the *shape* genuinely differs. If it
does match and `oflm-add` still refused, that is a bug in `find_open_kernels` or in
`spec_from_model_dir` — find it there, not in a table.

## A fine-tune is NOT a new shape — never give it a spec, a file, or a name

**A distilled, merged, or fine-tuned model that is architecturally identical to one
we already support is the normal case and must already work.** It is not a gap,
not a new model, and it gets **no file of any kind**.

- `Qwen3.8-Distilled-9B-NPU2` is exactly `Qwen3.5-9B` plus a fine-tune. Its derived
  `kernel_hash` is **identical** to `qwen35-9b.json`'s, and it binds to
  `Qwen3.5-9B-NPU2` today, sight-unseen, with nothing added for it. Verified.
- **Do not add a spec for a fine-tune.** `recipes/specs/` holds *shapes*, and a
  shape that already has a spec needs no second one. A per-finetune spec is a
  name in a table wearing a geometry costume — the exact failure mode this project
  exists to avoid. Adding one also costs a frozen hash in
  `test_lfm2.py::test_no_shipped_models_hash_moved`, which is the guardrail working.
- **A fine-tune that does not bind is a real finding, and the reason is worth
  reading before you add anything.** Ask which: same family, different *geometry*
  (a size nobody has built — `Qwen3.8-Distilled-4B` is hidden 2560 /
  intermediate 9216, a shape absent from the spec directory while `qwen35-9b` and
  `qwen35-27b` exist); or an unsupported family (`qwen4_exp`). The first is "build
  this shape once"; the second is "write a recipe". Neither is "add a file for this
  model".

The asymmetry that caused this: **`qwen35-27b` exists but there is no qwen35 spec at
hidden 2560.** A family got built at some sizes and not others. That is a real gap
and it is fixed by building the missing *shape* — once — never by special-casing the
model that asked for it.

## Packing a model does NOT build or ship its kernels — and never will

**This is the assumption that keeps reappearing, so state it plainly:** a user who
packs a model does not get kernels with it, and `oflm pack` is not going to start.

- Kernels are built **once per family**, on a machine that has the kernel toolchain
  and (for the BERT sets) an NPU, and they ship **with the application**. Building
  them per-model would mean compiling on whatever machine downloaded 90 GB of
  weights, which is exactly what `docs/BUILD.md` tells you not to do.
- So the kernel step is **not** part of a pack, and `--build-spec` only *reports*
  what would bind — it never compiles anything.
- **A model needs no `model_list.json` entry, no `xclbins/<its name>/` directory,
  and nothing else keyed to its name.** Those exist only for models whose *shape*
  has no kernel set. A fine-tune needs none of them.

The pack-time support audit reflects this. It reports whether an installed set
**binds this shape** (`_shape_kernel_set_installed`, `q4nx/cli.py`), the same match
`oflm add` performs — not whether a directory named after the model exists. An
audit that asks "is there something named after this model?" reports working
models as unsupported, and that is how people end up hard-coding.

When a pack reports no binding set, the reason is always one of two, and it is
worth telling them apart:

1. **The family has no kernels.** The recipe does not exist or `NOT_IMPLEMENTED`
   names it — e.g. `model_type 'qwen4_exp' has no recipe`. That is a kernel gap to
   close: a builder in `q4nx-build/q4nx/models/` plus a recipe. It is **not** about
   the model's size, its prune, or its name.
2. **The shape has no kernels built or shipped.** The recipe exists and validates,
   but no set for that geometry is installed. That is a packaging gap.

Both are fixed by building a shape that does not exist yet. Neither is fixed by an
entry, a special case, or a name lookup.

## Distribution: every open kernel that exists in the code ships

**A kernel set exists for every spec in `open_kernels/recipes/specs/`, and a
distribution contains all of them.** An open kernel not being installed is
acceptable for exactly one reason: it does not exist anywhere in the code. If a
recipe builds it, it is built and shipped — regardless of what the machine doing
the build happens to have.

Concretely, when you touch the build or release path:

- **`export-kernels.py` with no `--specs` builds every spec in the directory**
  (`spec_list("")` → `SPECS_DIR.glob("*.json")`, `:132`). Never narrow that to a
  hand-kept list. The spec directory is the source of truth; a name typed
  anywhere else is a list that will drift.
- **A failed spec must be loud.** `export_open_kernels` counts failures and `main`
  returns non-zero (`:144-149`, `:244-246`), which fails the `export_kernels`
  CMake target (`src/CMakeLists.txt:1355`). Do not add a "continue on failure" or
  a warning-only path. A distribution missing a kernel must not be releasable.
- **`--skip-specs` and `--skip-bert` are for a developer's local iteration, not
  for a release.** `nix/open-kernels.nix:28` exposes `skipSpecs`; it exists so a
  Nix builder can omit a recipe it cannot satisfy, and a release derivation must
  pass `""`. If you find `skipSpecs` set on the release derivation, that is the
  bug to fix, not a limit to document.
- **BERT sets need an NPU at build time, which is a machine property, not an
  existence property.** A release is staged on an NPU machine
  (`utilities/release/stage-prebuilts.sh`); that is how the sets get into
  `prebuilts/`. `verify-prebuilts.py` then refuses a release that has no
  `platforms.linux` section, so a release cannot quietly ship without them.
- **The check that matters at the end**: a user who has *any* model whose shape
  has a recipe must find its kernels already installed. That is the whole
  promise. It fails today only if a spec silently did not build.

On `Ornith1.5-9B`-style names: a name that is not a shipped model is exactly what
the tools handle. It is **not** expected to match `Qwen3.5-9B`'s kernel set, and
it should not — it is a different shape. It gets kernels because its *shape* has a
recipe, or it does not run on open kernels yet, which is a gap to close. What it
must never need is an entry keyed to its name.

### Padding and pruning exist so a model lands ON an existing kernel set

`q4nx-build` can alter a model's geometry, and **the point of both mechanisms is
to reach a shape that already has kernels** — not to mint a new one:

- **`--pad-to-fit`** zero-pads the hidden axis up to a compiled variant's width.
- **`--prune-ffn K` / `--prune-moe-ffn K` / `--prune-experts K`** narrow a real
  axis, ranked by **imatrix activation energy** (top K, not first K). Measured on
  the Qwen3.8-27B imatrix at K=12288 of 17408: **88.5% mean activation mass
  retained vs 70.6% for a tail chop**. It refuses without an imatrix rather than
  chopping blind, and `ffn_index_set` derives ONE index set per layer from
  `ffn_down`'s scores so all three tensors select the same axis — per-tensor
  pruning loads cleanly and computes a different network, silently.
- **Every packed tensor is padded to the container's block size** (`row_block_size`
  32, `col_block_size` 256 — `model_converter.py:809-820`, and the configs under
  `q4nx-build/configs/`). So a pruned width lands on a block multiple, and
  **pruning and padding both push a model toward a geometry that already exists.**

**So do not treat a padded or pruned container as needing its own kernels.** It
needs the set built for the width it actually reaches — which is very often one
that is already installed, and is found by the ordinary shape match, with no
special case. `q4nx-build` records the true width in the container's
`config.json` (`intermediate_size` → the kept count, plus an `oflm_pruned_ffn`
note with method and retained mass — `model_assets.py:1405-1417`), which is
exactly what lets that match happen honestly.

**If you find yourself adding a gate, a special case, or a name lookup to make a
padded/pruned model run, that is backwards.** The mechanism exists so the ordinary
shape match finds a set that is already there. Pruning is a *quality* trade the
container labels for the user (`pruned_note()`), not a licensing or binding
condition.

### After changing the manifest schema, re-export every spec

A `manifest.json` written by an older recipe carries only `spec_hash`, and an
already-installed kernel set keeps that file until it is re-exported. Such a set
binds only a byte-identical spec — it cannot bind a shape-identical fine-tune,
because nothing in it records the shape key. So:

- **Adding or changing a manifest key means re-exporting the kernel sets** that
  consumers depend on, and reinstalling. `utilities/build-all.sh` (or
  `--force` on `export-qwen36_kernels.py`) does all 23.
- **`specs/open-engine/tests/fixtures/*.json` are checked-in manifests.** After
  any `manifest.py` change run `python specs/open-engine/tests/make_fixtures.py`,
  or `test_manifest_fixture.py` fails. (It checks that the fixture is the recipe's
  current output — a fixture is a manifest, so a schema change invalidates it.)
- **Diagnose "it says no kernels" by checking for the key first**, not by
  rebuilding. `python -c "import json;print('kernel_hash' in json.load(open('<set>/open_kernels/manifest.json')))"`
  — if it is missing, the set is stale and needs a re-export, not a new kernel.

## Rules

- **Never type a model name into a table to make something load.** If a model is
  refused, you have found a missing `ModelSpec` field, a missing builder, or the
  `real_vocab` hash bug above. The refusal is information.
- **The refusals in the code are mostly real.** "rope_scaling type X is not
  supported", "expert intermediate equals hidden" — those describe shapes the
  kernels genuinely cannot compute, and `recipes/families.py::NOT_IMPLEMENTED`
  names each gap with the reason inline. Keep those. The one that does *not* fit
  the project's purpose is a hash that separates two models with identical kernels.
- **Closed requirements get "not implemented" messaging first.** If you find
  yourself needing a closed binary to finish a path, add the message, then work on
  the open replacement. Do not wire the closed one in.
- **A `layer_types` member is free; a `ModelSpec` field is not.** Adding a member to
  `LAYER_TYPES` (`spec.py:19-23`) moves only the specs that use it. Adding a
  dataclass field moves **every** hash in the tree and invalidates every shipped
  kernel set. Say so before you add one.
- **A shape with no open kernels is a gap to close, not a model to exclude.** Add the
  builder to `q4nx-build/q4nx/models/` and the recipe, then write the skill.
- **Keep `q4nx-build` in sync with the formats models actually ship.** It should
  never require a manual step; if one is unavoidable, the user must be told the
  exact command.

## Trust order

There is no single source of truth in this repo, and that is the known problem.
When sources disagree, believe them in this order:

1. **`oflm-add` and `q4nx-build`** — what they expect and build.
2. **The code** — `open_kernels/recipes/`, `utilities/oflm-add/oflm_add/`,
   `utilities/q4nx-build/q4nx/`.
3. **The skills** under `.opencode/skill/` — accurate in the open-kernel areas,
   stale in the older ones. Verify a path before you rely on it.
4. **Everything else** — `README.md`, `docs/`, `src/create_new_model.md`,
   `REPLACEMENT_PLAN.md`, `PR_open_npue.md`. Largely stale; several describe the
   pre-recipe mental model or the closed-engine path. Useful as history, not as
   instructions. `open_kernels/README.md` is a vendored log with paths
   (`C:/code/phlegm`) that do not exist here.

Documented traps, verified: several skills point at `.claude/plans/*` (gitignored,
absent) and `docs/ExampleNPU/` (deleted). `src/README.md` predates the `flm`→`oflm`
rename and references three scripts that no longer exist.

**Never hard-code a spec count.** Count the directory:
`ls open_kernels/recipes/specs/*.json | wc -l`. The count has been stale in three
places at once before (docs said 12, `nix/open-kernels.nix` said 13, reality was
15, then 23). If you add or remove a spec, do not go hunting for prose to update —
just make sure nothing states a number, or states it from the directory.

## Model formats and where new ones go

New Q4NX models live on HuggingFace as `Atomic-Germ/*-nB-OpenNPU2`. Each needs a
builder in `q4nx-build/q4nx/models/` as it gains xclbin support. **Family xclbins
ship with the application**; a converted model ships with the rest and links to its
family xclbin regardless of fine-tuning or derivation. This is what lets new
shape-identical models be added with no code change, distributed through
`oflm-add`.

## Working notes

- **`utilities/oflm-test`** runs the full suite: `--llm`, `--vision`, `--embed`,
  `--tools`. Run it before claiming anything works.
- **Put tools you write in `utilities/`**, not a temp folder.
- **After each successfully packed model, create or update a skill** so the next
  agent doesn't repeat the research.
- **Toolchain:** activate `ironvenv/bin/activate`. Peano (llvm-aie) is deliberately
  *not* on `PATH` to avoid clashing with system clang — it lives under
  `./ironvenv/lib/python3.*/site-packages/llvm-aie/bin`. Check which minor version
  your venv carries with `ls ironvenv/lib` rather than assuming; this checkout is
  `python3.11`. Source `utilities/mlir-aie/utils/env_setup.sh` if needed.
- You can install or request toolchain components.

## Skills

All 21 live in `.opencode/skill/`. **Check this list before you start** — the open
kernels are not one subsystem, and an agent that only ever loads the first relevant
skill will miss most of the project. The set spans dense/MoE LLM kernels, GGUF
direct, embedding and Whisper, the FLUX diffusion stack, WideDeltaNet, and
packaging/release.

<details>
<summary>Full skill index (21)</summary>

- `open-dense-kernels` — the dense recipe (`recipes/dense.py`): qwen2/qwen3/llama3/
  gemma/lfm2, and the `dx` / `dx_attn` attention designs.
- `open-gguf-direct` — loading GGUF straight into the engine, `gguf_pool.py`.
- `open-qwen36-kernels` — the Qwen3.6-MoE engine kernels (lx/ax/ln/lm_head_q8).
- `open-qwen38-27b-kernels` — Qwen3.8-27B on `recipes/qwen35.py` at hidden 5120.
- `open-granite-kernels` — Granite 4.2 3B on the dense recipe.
- `open-phi3-nanbeige-kernels` — Phi-4-mini (phi3) and Nanbeige4.1-3B (llama3).
- `open-wide-deltanet-ab` — the banked alpha/beta dispatch at H5120/H2560.
- `open-wide-deltanet-chain` — the AB, conv/record and recurrent-state chain.
- `open-attention-row-groups` — odd row-block counts in attention GEMM builds.
- `dit-gemm` — the FLUX.2 diffusion GEMM kernel set.
- `dit-fa` — the diffusion flash-attention kernel (d=128).
- `dit-ew` — the diffusion elementwise / row ops.
- `dit-conv` — the diffusion implicit-GEMM conv and the VAE decoder schedule.
- `open-diffusion` — the whole-image FLUX.2 [klein] 4B schedule and pyxrt runner.
- `open-diffusion-edits` — the VAE encoder and the image-edit path.
- `full-elf-context` — several kernel sets in one hardware context.
- `npu-offload-pipeline` — pointer to the long-form offload playbook.
- `openflowlm-packaging` — RPM/TGZ/DEB/MSI builds and install.
- `openflowlm-release` — tag-triggered releases and prebuilt kernel staging.
- `openflowlm-nix-packaging` — the flake and the NixOS module.

Plus `npu_offload_pipeline.md`, a long-form playbook. **It is stale** — it is
written against a deleted `docs/ExampleNPU/` tree and models the project as
"enumerate a model's GEMMs and write a new IRON design". Under the recipe model
that is the answer only for a genuinely new *shape*. For a shape-identical
fine-tune the answer is zero kernel work.

</details>