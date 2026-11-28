---
title: "Kernel Contributions"
description: "Build, verify, and register open MLIR-AIE and IRON kernel sets for new model sizes and families."
icon: "memory"
weight: 30
---

`open_kernels/` holds OpenFlowLM's MLIR-AIE / IRON designs compiled into `.xclbin` and `.elf` artifacts alongside a generated `manifest.json` interpreted by `src/open_qwen36`.

> [!TIP]
> Read [`.agents/skills/open-kernel-pipeline/SKILL.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/.agents/skills/open-kernel-pipeline/SKILL.md) and [`.agents/skills/open-kernel-families/SKILL.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/.agents/skills/open-kernel-families/SKILL.md) for recipe internals and tiling constraints.

## Geometry-Driven Pipeline

Tensor dimensions define a kernel set rather than repository names. Fine-tunes reuse their base model's compiled kernel set with zero rebuilds; new sizes build via one command; only new layer arithmetic requires recipe or operator code.

```text
weights (GGUF / config.json)
  -> geometry -> family recipe -> spec (open_kernels/recipes/specs/*.json)
  -> kernel set (src/xclbins/<set>/open_kernels/) -> <model>/open_kernels symlink
```

| Artifact | Path | Maintained By |
|---|---|---|
| Spec | `open_kernels/recipes/specs/*.json` | `gen-kernel-spec.py` / `oflm add` |
| Recipe | `open_kernels/recipes/{dense,qwen35,qwen36moe,lfm2}.py` | Kernel author |
| Design | `open_kernels/designs/<op>/` | Kernel author |
| Catalogue | `open_kernels/recipes/catalogue.py` | `catalogue-add-point.py` after NPU verification |
| Kernel set & Manifest | `src/xclbins/<set>/open_kernels/` | `export-kernels.py` (gitignored) |

---

## Adding a Size vs. Adding a Family

<div class="oflm-two-col">
  <div class="oflm-card">

### New Model Size (`build-new-spec.py`)

Derives the spec, compiles unvalidated kernels, verifies operators on XDNA2 hardware against FP64 references, updates `catalogue.py`, rebuilds cleanly, and updates frozen test hashes:

```bash
python utilities/build-new-spec.py --repo Qwen/Qwen3-8B
# Local GGUF:
python utilities/gen-kernel-spec.py --gguf model.gguf \
  --out open_kernels/recipes/specs/<name>.json
python utilities/build-new-spec.py --spec <name>
```

  </div>
  <div class="oflm-card">

### New Architecture Family

1. Map the architecture in `open_kernels/recipes/spec.py` (`GGUF_FAMILIES` + HF config).
2. Add a recipe module under `open_kernels/recipes/` and register in `FAMILIES` (or `NOT_IMPLEMENTED` if unimplemented).
3. Reuse `designs/dense/dx.py` or `designs/layer_x`, adding new designs only for new operators.
4. Verify on XDNA2 hardware and record findings in `open-kernel-families/SKILL.md`.

  </div>
</div>

---

## Toolchain & Verification

- **Linux:** `source utilities/env.sh` sets XRT, ROCm/HIP, `mlir-aie`, Peano (`PEANO_INSTALL_DIR`), and `OFLM_*` paths.
- **Nix / NixOS:** `nix develop .#open-kernels` ([Nix & NixOS](/docs/install/linux/nix/)).
- **Windows 11:** Check out `mlir-aie` at tag `v1.4.2` and run `iron_setup.py` ([Build System](/docs/contributing/build/)).

> [!WARNING]
> Never run two kernel exports concurrently in the same checkout (`designs/layer_x` regenerates C++ translation units per spec). Use `git worktree` for parallel comparisons.

Verify end-to-end before opening a PR:

```bash
oflm check <tag> && oflm run <tag>
oflm-test --llm --model <tag>
python utilities/export-kernels.py --check <baseline-dir>
utilities/check-settings-ci.sh
```

| Diagnostic | Resolution |
|---|---|
| `no open kernel set can run this model` | Run from a source checkout so `oflm add` builds the set, or update release package |
| `this build of the set predates GGUF-direct` | Re-export with `python utilities/export-kernels.py` |
| `the recipe cannot build this shape` | Spec saved in `recipes/candidates/`; requires tiling/design update |
| `the catalogue has not validated N point(s)` | Run `build-new-spec.py` on XDNA2 hardware (never ship `OPEN_KERNELS_UNVALIDATED=1`) |
| `no open kernels found` | Re-run `oflm add` or check `OFLM_OPEN_KERNELS_DIR` |
