# Developer Workflow (`oflm dev`)

This document describes the practical, shape-first workflow for developing and extending OpenFlowLM's open NPU kernel support: probing model geometries, filling missing sizes, validating catalogue points, and exporting kernel sets.

**Read first:** [Kernel contributions](kernel-contributions.md), the [open-kernel pipeline](../../.agents/skills/open-kernel-pipeline/SKILL.md), and [open-kernel families](../../.agents/skills/open-kernel-families/SKILL.md). This guide focuses on day-to-day tooling and the `oflm dev` commands that wrap it.

---

## Core idea: geometry is the identity

A model is the numbers in its weights (hidden size, layers, heads, head_dim, rotary, vocab, FFN/layout), not its repo name, fine-tune label, or quant suffix.

- `oflm add` reads the weights (GGUF header is authoritative; config/container for other sources), derives `family + geometry spec`, and resolves a kernel set by **spec/manifest geometry** (including the `gguf` manifest view for GGUF-direct).
- **Same shape ⇒ same kernels.** Fine-tunes, renames, and many quant variants relink/reuse an existing set. Depth can be immaterial when the kernelized width/layout matches.
- **New size of a supported family** is usually spec + build + gate + export work, not new kernel code.
- **New architecture** is recipe/design work.
- “Runs/links” is not the same as “fully supported/recommended”. The project aims to allow compatible shapes to run, but production support requires validation and measurement.

---

## Prerequisites

Kernel development requires:

- **XRT** and the AMD XDNA/NPU driver (`/dev/accel/accel0` on Linux)
- **mlir-aie + Peano** (IRON toolchain), typically via `ironvenv`
- **NPU hardware** for GATE/validation steps (compilation can run without it; validation cannot)
- **Source/build environment** for `oflm dev` wrappers that invoke `utilities/` and `open_kernels/` scripts

### In a source tree

```bash
# Full toolchain + build workflow (load-bearing)
cmake --workflow --preset linux-default

# Or load environment for dev scripts/utilities
source utilities/env.sh
# utilities/env.sh --check
```

`cmake --workflow --preset linux-default` (and related presets) handles toolchain setup, XRT discovery, and kernel export wiring. For fast engine-only iteration, use debug presets with kernels off as documented in [BUILD.md](../BUILD.md).

### From packages

`oflm dev` runs from an OpenFlowLM source checkout: it locates the tree by walking up from the working directory (or the executable) to `utilities/build-new-spec.py`, and `OFLM_REPO_ROOT` can point at one explicitly. It warns when required toolchain components are missing and points to this path.

Check status:

```bash
oflm dev env-check
```

---

## `oflm dev` commands

`oflm dev` wraps common source-tree developer workflows (auto-detects repo root; override with `OFLM_REPO_ROOT`):

| Command | Purpose | Wraps / notes |
|---|---|---|
| `oflm dev build-new-spec [args...]` | Derive/build/gate/catalogue/register a new geometry | `utilities/build-new-spec.py` |
| `oflm dev probe-sizes [args...]` | Probe GGUF repos for `served/would_build/unreadable/...` | `utilities/probe-family-sizes.py` |
| `oflm dev export-kernels [args...]` | Export/build open kernel xclbin sets | `utilities/export-kernels.py` |
| `oflm dev op-gate [args...]` | Run fixtures to validate missing catalogue points | `utilities/op-gate.py` |
| `oflm dev gen-kernel-spec [args...]` | Derive a candidate geometry spec from a repo/GGUF | `utilities/gen-kernel-spec.py` |
| `oflm dev gen-base-patch [args...]` | Populate `recipes/patches.json` from a repo's base chain (`HF_TOKEN` forwarded for gated bases) | `utilities/gen-base-patch.py` |
| `oflm dev spec-gaps [args...]` | Report registry/checkout sizes with no kernel spec | `utilities/spec-gaps.py` |
| `oflm dev validate-spec [args...]` | Check a spec's geometry/export name before it ships | `utilities/register-spec.py` |
| `oflm dev add-point [args...]` | Record a validated point in the op catalogue | `utilities/catalogue-add-point.py` |
| `oflm dev refresh-model-info [args...]` | Refresh `src/model_info.json` hashes/sizes from HuggingFace | `utilities/refresh-model-info.py` |
| `oflm dev family-doctor` | Check the machine-verifiable family/registration invariants (no NPU needed) | `utilities/family-doctor.py` |
| `oflm dev verify-family [args...]` | Run NPU-free model checks + a production-bar report honestly marked `needs-measurement` | `utilities/verify-family.py` |
| `oflm dev add-family --repo <user>/<repo>` | Make a repo installable by geometry, doing the least work: report "already served" if a set matches, delegate a new size to `build-new-spec --repo` (which parks failures in `candidates/` without registry edits), or register a new family with refusing arithmetic stubs verified by `family-doctor`. Advanced `--name <key>` registers directly. | `utilities/add-family.py` |
| `oflm dev run-kernel <cfg> [args...]` | Run a `.cfg` fixture on NPU via `open_kernels/harness/run_kernel` (builds harness if needed; `OFLM_RUN_KERNEL` overrides) | `open_kernels/harness` |
| `oflm dev clean-specs [--dry-run]` | Remove failed/candidate specs in `open_kernels/recipes/candidates/` | built-in |
| `oflm dev clean-kernels [--sets]` | Remove kernel export scratch (and optionally `src/xclbins/*`) | `utilities/clean-kernels.py` |
| `oflm dev env-check` | Show toolchain/environment status | built-in |

Aliases like `--build-new-spec`, `--probe-sizes`, `--export-kernels`, `--op-gate`, `--clean-specs`, `--env-check`, `new-spec`, `run-kernel`/`harness`/`run_kernel`, `gen-spec`, `base-patch`, `check-spec`/`register-spec`, `add-point`/`catalogue-add-point`, `refresh-info`, `spec-gaps`, and `family-doctor` are accepted.

The commands above wrap the whole spec → gate → catalogue → rebuild → validate flow, including spec generation from a GGUF-only repo (`oflm dev build-new-spec --repo <user>/<repo>`, or `--gguf-file <local.gguf>`). You do not need to run the underlying Python scripts by hand; their options are passed through unchanged by each `oflm dev` subcommand. Specialized exports in `open_kernels/` (e.g. `export_whisper_kernels.py`, `export_dit_kernels.py`, `compose_elf.py`) and `open_embedding/` tooling are still separate.

---

## Typical workflow: fill a missing size (supported family)

### 1. Probe for geometry gaps

Prefer `mradermacher/*-i1-GGUF` (often `Q4_1`) and fall back to `unsloth/*-GGUF`.

```bash
# Built-in sweep
oflm dev probe-sizes --builtin --jobs 4 --timeout 120 --json /tmp/oflm-size-probe.json

# Specific candidates
oflm dev probe-sizes --repo mradermacher/Qwen2.5-7B-Instruct-i1-GGUF \
                     --repo unsloth/Qwen2.5-7B-Instruct-GGUF
```

Interpretation:
- **`served`**: a matching open-kernel set already serves this geometry (install/link/reuse).
- **`would_build`**: supported family/recipe, different geometry — candidate to fill.
- **`unreadable` / no recipe / geometry unreadable**: arch/recipe mapping or metadata issue; verify the actual model arch/family (similarly named models may differ, e.g. Granite 3.x vs 4.2, original `qwen3_moe` vs covered qwen3_next/qwen3.5-moe variants).
- **design wall** (often surfaced during spec/build preflight): geometry cannot be built by the current recipe/design (tile/k-width constraints, split-FF limits, etc.). These belong in `open_kernels/recipes/candidates/` with reasons.

Always trust derived geometry/spec/hash and `--dry-run` over model names.

### 2. Dry-run the spec/build plan

```bash
oflm dev build-new-spec --dry-run --repo mradermacher/Qwen2.5-7B-Instruct-i1-GGUF
# a gated/offline repo with a local GGUF:
# oflm dev build-new-spec --dry-run --gguf-file model.gguf --repo <base-repo>
# an already-generated spec:
# oflm dev build-new-spec --dry-run --spec <name>
```

The spec name and export label come from the **base model** the repo quantizes,
not from the upload slug (`unsloth/Llama-3.2-1B-Instruct-GGUF` and
`mradermacher/Llama-3.2-1B-Instruct-i1-GGUF` resolve to the same base). Geometry
matching at install time uses `kernel_hash()` / `serves()`, so a user can install
**any tune of a supported base** with `oflm add <user>/<repo>` and get the right
kernel set without a name matching, a chat-template match, or any other manual
step. Depth and parameter count are not part of that match: a 27B set runs a
40B merged model at a different depth.

This reports recipe walls and missing catalogue points (attention tuple, op fixtures like `gemv_q4`, `ln`, `lm_head_q4`, etc.).

### 3. Build, gate, and validate (NPU required for GATE)

```bash
oflm dev build-new-spec --repo mradermacher/Qwen2.5-7B-Instruct-i1-GGUF
# or: oflm dev build-new-spec --spec <name>
```

`build-new-spec.py` stages:
1. **SPEC** — derive/link geometry
2. **BUILD** — compile with `OPEN_KERNELS_UNVALIDATED=1` to prove the design compiles
3. **GATE** — run fixtures/oracles on NPU for missing points (only points that pass are added)
4. **CATALOGUE** — record validated points
5. **REBUILD** — rebuild through validated path
6. **VALIDATE** — check the geometry signature and export label; no files are edited

Commit only the generated **spec JSON** and the validated **catalogue points** —
there is no hash-locked test file or name list to update. Users on the new
version then install any tune of that base with `oflm add` and it Just Works.

Notes:
- Nothing here needs the model’s weights, a converted container, or the engine.
- Some geometries fail due to **design walls** or **fixture limitations** (not the same thing).
- `op-gate.py` retries valid `per_call` divisors for `gemv_q4` gates when a fixture fails to fit/build at a given `per_call`.

### 4. Clean failed candidates / scratch (optional)

```bash
oflm dev clean-specs --dry-run
oflm dev clean-specs

oflm dev clean-kernels          # remove export scratch
oflm dev clean-kernels --sets   # also remove exported sets in src/xclbins (full re-export)
```

### 5. Install and validate

```bash
oflm add mradermacher/<model>-i1-GGUF
oflm run <tag>
# oflm check <tag> / utilities/oflm-test ... as applicable
```

For “fully supported” claims, follow the project’s production bar (correct registry/manifest behavior, measured prefill/decode vs context, NPU coverage, features exposed, etc.). For v0.1.0-open goals, prioritize filling geometries of supported families and making compatible shapes runnable by geometry.

---

## Exporting kernels

```bash
# all families/sets (dense specs + BERT embedding sets + diffusion sets, as configured)
oflm dev export-kernels

# specific dense/open_kernels spec(s) only
oflm dev export-kernels --specs qwen3-8b --skip-bert --skip-diffusion

# skip known-bad specs in a broad run
oflm dev export-kernels --skip-specs hy-mt1.5-1.8b-i1-gguf,qwen2.5-1.5b-instruct-i1-gguf

# tune parallelism if toolchain/NPU contention causes flaky failures
oflm dev export-kernels -j 4
```

Notes:
- `--specs` filters **open_kernels/dense spec exports**; BERT and diffusion families still run unless `--skip-bert` / `--skip-diffusion` are set (use `--diffusion-only`, `--bert-only`, or `--whisper-only` for focused runs).
- Whisper kernel export is separate: `oflm dev export-kernels --whisper` (or `--whisper-only`, `--whisper-no-fa`) wraps `open_kernels/export_whisper_kernels.py`.
- Intermittent failures can happen under heavy parallel build/load; rerunning (or lowering `-j`) often isolates real breakage from contention/flakes.
- See [BUILD.md](../BUILD.md) for preset-driven exports and packaging (`linux-default`, `linux-package`, etc.). `export-kernels.py` coordinates kernel families and toolchain/venv requirements.

---

## Large-K / fixture validation gaps (current frontier)

Some geometries (e.g. large-K `gemv_q4` points such as K=13824 in Qwen2.5-14B, and similar 32B-class points) can pass recipe/shape checks but fail **standalone gate fixtures** due to AIE tile L1/layout constraints in the fixture design, even after exhausting valid `per_call` divisors.

For `gemv_q4` fixtures, per compute-tile resident footprint is dominated by:
- activation table `tab` (~2.25*K)
- broadcast/input activation buffer `x` (~2*K)
- double-buffered weight elements `w` (~2 * `per_call` * 5120 B for q4nx chunks)
- stack/output buffers

At large K this can exceed a ~64 KB tile L1 budget in the current standalone fixture dataflow (e.g. K=13824, `per_call=1` still over budget), so the failure is a **fixture/standalone-design layout limit**, not host/system RAM and not automatically “model cannot run”.

**Next-move directions (empirical):**
- **Large-K gate/fixture redesign:** stream or K-tile `x`/`tab` (and/or adjust buffering) so validation can fit while still exercising the target point; may require a dedicated gate path/variant in `open_kernels/designs/gemv_q4`.
- **Design/layout exploration:** evaluate core count/placement, buffer lifetimes, and memory reuse in the fixture context; document what is fixture-only vs production-relevant.
- **Alternate validation path for unfittable standalone fixtures:** where safe and explicit, validate kernel math/geometry through a different but equivalent harness or composed context, while keeping catalogue claims honest.

When this occurs, capture the point `(K, rs, rows_per_core, per_call)`, the memory map/error, and the attempted divisors; treat as a validation gap to close rather than a silent skip.

---

## Practical tips

- **Use `--dry-run` early and often** (`oflm add --dry-run`, `build-new-spec --dry-run`, `probe-sizes`).
- **Do not guess mappings:** original `qwen3_moe` vs `qwen3_next`/qwen3.5-moe, Granite 3.x vs 4.2, Gemma4 unified vs unsupported sub-variants, etc. are different contracts.
- **Shape reuse is the feature:** many derivatives/distills/size-label variants resolve to existing sets (e.g. phi4-mini/phi3 geometry cases, LFM2.5 on LFM2 geometry, qwen3.5/3.6-moe class reuse, depth-immaterial cases where kernelized geometry matches).
- **Never `git add -A` / `git add .`** in this repo; stage specific paths and review `git diff --cached --stat`.
- **`build-new-spec.py` commits only the spec JSON and catalogue points** — the geometry is the contract; no frozen hash or name list is edited.

---

## Where things live

| Area | Path |
|---|---|
| Specs (geometry) | `open_kernels/recipes/specs/*.json` |
| Candidate/wall specs | `open_kernels/recipes/candidates/` |
| Recipes/families | `open_kernels/recipes/` |
| Designs (ops/kernels) | `open_kernels/designs/` |
| Catalogue (validated points) | `open_kernels/recipes/catalogue.py` |
| Exported kernel sets (build output) | `src/xclbins/<set>/open_kernels/` (not committed) |
| Dev utilities | `utilities/` |
| Pipeline/family skills | `.agents/skills/open-kernel-pipeline/`, `.agents/skills/open-kernel-families/` |
