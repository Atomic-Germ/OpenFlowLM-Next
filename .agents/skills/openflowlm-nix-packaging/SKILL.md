---
name: openflowlm-nix-packaging
description: Build, validate and debug the Nix flake and NixOS module that package the OpenFlowLM engine and the open NPU kernel xclbins (flake.nix, nix/, NIX.md). Use when touching nix/package.nix, nix/open-kernels*.nix, nix/nixos-module.nix, nix/shell.nix, when a `nix build .#oflm` / `.#openflowlm-open-kernels` fails or produces an oflm that cannot find its kernels, when "xclbins not found" happens under Nix, or when a new model family needs a Nix kernel package.
---

# Nix packaging for OpenFlowLM

`flake.nix` + `nix/` package the engine (`.#oflm`), the open NPU kernel xclbins
(`.#openflowlm-open-kernels`), two dev shells, and a NixOS module
(`nixosModules.default`). Cherry-picked from eyduh's fork (issue #110) and
validated end to end on an XDNA2 laptop: Nix-built engine + Nix-built kernels +
Nix XRT ran real inference.

Read `NIX.md` first — it is the user-facing doc and must stay true to the code.

## Ground truth from a validated build

| Step | Wall time (12 cores) | Notes |
|---|---|---|
| `nix build .#openflowlm-open-kernels` | ~10 min | all 12 specs, mlir-aie 1.4.3 + Peano 22.0.0 |
| `nix build .#oflm` | ~15 min | includes the kernels above; `doCheck` runs 4 ctest tests |

## Setting up Nix without root

Fedora ships `nix` but its store is not writable. Use `nix-portable`:

```bash
curl -sL -o ~/.local/bin/nix-portable \
  https://github.com/DavHau/nix-portable/releases/download/v012/nix-portable-x86_64
chmod +x ~/.local/bin/nix-portable
export NP_GIT=/usr/bin/git            # else it bootstraps its own git
~/.local/bin/nix-portable nix --extra-experimental-features 'nix-command flakes' ...
```

**Set `TMPDIR` to the real disk.** Nix unpack steps write there, and a small or
full `/tmp` tmpfs fails with a misleading `cp: Disk quota exceeded` while
unpacking XRT:

```bash
mkdir -p ~/nixbuild-tmp && export TMPDIR=~/nixbuild-tmp
```

The store lives in `~/.nix-portable` (~36 GB for a full build). Watch `df -h /`;
`nix store gc` inside nix-portable reclaims it. Everything a build produces is
reproducible, so GC freely between experiments.

## The four traps in this packaging (all fixed once already — do not reintroduce)

1. **A derivation cannot write into its source tree.** Both kernel exports write
   `open_kernels/designs/*/build` and `src/xclbins` *inside the source*. The
   store is read-only, so `nix/open-kernels.nix` copies the source to
   `$TMPDIR/openflowlm` first and builds there.
2. **`src/xclbins` is git-ignored, so a flake source built from a working tree
   carries the developer's local kernels.** `nix/open-kernels.nix` and
   `nix/package.nix` both `rm -rf src/xclbins` (the latter in `postPatch`,
   because CMake installs `xclbins` when it exists). Without this the store
   shipped whatever happened to be in the checkout.
3. **`ln -sfn target dir` does not replace a directory** — it creates the link
   *inside* it. `nix/package.nix` does `rm -rf` then `ln -s`.
4. **A NixOS module has no `self` argument.** `imports = [ self.inputs.… ]`
   resolves `self` through `_module.args`, which requires `config`, which is
   evaluating `imports` — infinite recursion, and the eval fails long before
   anything is built. The nix-amd-ai import therefore lives in `flake.nix`,
   where `inputs` is in scope. Verify the module through a **separate consumer
   flake**; evaluating it inside this repo's own flake gives false results.

## How the engine finds kernels

`src/common/utils.cpp` searches, in order: `$OFLM_XCLBIN_PATH`,
`$OFLM_CONFIG_PATH`'s directory, `~/.config/oflm`, `~/.flm`, then
`<exe_dir>/../share/oflm`. Only one `OFLM_XCLBIN_PATH` is allowed, and it may
point into the read-only store, so:

- The kernels are symlinked into the **engine package's own**
  `share/oflm/xclbins` — a root that needs no environment variable.
- The `oflm` launcher defaults `OFLM_CONFIG_PATH`/`OFLM_MODELINFO_PATH` to
  `$HOME/.config/oflm` with `${VAR:-…}` (never plain assignment, so a caller's
  export wins) and deliberately does **not** set `OFLM_XCLBIN_PATH`;
  `oflm-add` writes model kernels to `~/.config/oflm/xclbins` for the same
  reason (`user_xclbin_dir` ignores that variable too).

The NixOS module merges `kernelsPackage` + `extraXclbinPackages` with
`symlinkJoin` and `overrideAttrs`es the engine package's `postInstall` to point
`share/oflm/xclbins` at the merge.

## CMake flags the derivation must set

- `-DOFLM_INSTALL_PATH_PLUMBING=OFF` — otherwise the install writes
  `/etc/profile.d/openflowlm.sh` and `/usr/bin/oflm` **outside** `$out`, which
  fails in the sandbox.
- `-DOFLM_BUILD_KERNELS=OFF` — kernels are their own derivation.
- XRT's plugin loader resolves `libxrt_driver_xdna` next to `libxrt_core`, so
  plain `pkgs.xrt` cannot enumerate the NPU. Everything (package, dev shells,
  kernel build) uses an `xrt-combined` runCommand that symlinks
  `xrt-plugin-amdxdna` into `$out/lib`.
- `third_party/tokenizers-cpp` is a submodule: a flake source has it empty, so
  `postPatch` fetches it at the pinned revision and injects
  `nix/tokenizers-cpp-cargo.lock`.

## The bundled Python utilities

`oflm-test`, `q4nx-build` and `oflm-add.py` ship in the package and their
interpreters are repointed at a Nix Python (a bare `python3` does not exist on
NixOS). Their **pip dependencies are not vendored** — `openai` for `oflm-test`,
`gguf numpy torch einops safetensors huggingface-hub` for `q4nx-build`. Do not
try to vendor them with `python.withPackages`: nixpkgs' matplotlib/jupyter test
suites fail in constrained builders and fixing that needs a nixpkgs-wide
`doCheck = false` override. `NIX.md` documents a venv instead, and the wrapper
uses `--suffix PATH` so a user's environment wins over the packaged Python.

`modelscope` is only a fallback to AMD's re-uploads and is insecure in nixpkgs
(CVE-2026-84202, unsafe YAML in model config loading); leave it out.

## Validation, in increasing order of cost

```bash
nix flake show                                     # everything evaluates
nix build .#openflowlm-open-kernels               # all specs compile
nix build .#oflm                                  # engine + ctest (4 tests)
<out>/bin/oflm validate                           # NPU seen through the nix XRT
```

End to end, on an NPU host, with the smallest model:

```bash
OFLM_EXECUTABLE=<out>/bin/.oflm-wrapped \
  <out>/share/oflm/oflm-add/oflm-add.py --tag lfm2:1.2b Atomic-Germ/LFM2-1.2B-NPU2
<out>/bin/oflm run lfm2:1.2b
```

`OFLM_EXECUTABLE` matters: `oflm-add` finds kernels relative to `oflm` on
PATH, so without it a system install at `/opt/openflowlm` wins over the Nix
store and you silently test the RPM's kernels. `oflm-add` prints the link it
made — check it points into `/nix/store`.

To test one spec instead of all twelve, call the derivation with a skip list:

```bash
nix build --impure --expr '(import <nixpkgs-with-overlay> {}).callPackage ./nix/open-kernels.nix {
  srcRoot = ./.; skipBert = true;
  skipSpecs = "granite42-3b,qwen3-4b,…";   # everything except the one you want
}'
```

`skipSpecs` defaults to empty because the three specs that used to need
skipping (`qwen25-3b`, `minicpm5-2b`, `phi4-mini-4b`) build again after the
recipe fixes. Re-add a name if one regresses.

## Kernel dev shell side effects

`.#open-kernels` materializes a 1.3 GB `ironvenv/` **inside the checkout**
(git-ignored) because the export scripts need a writable venv, and the venv's
`bin/python` is re-pointed at the shell's Python on every entry. Delete it when
you are done. `nix-shell nix/shell.nix` is the same shell without flakes.

## Adding a model family's kernels

Nothing Nix-specific: a spec in `open_kernels/recipes/specs/` is picked up by
`export-kernels.py`, which builds every spec it finds. `extra.model` in that
spec is the export directory name **and** the name `oflm add` links by — two
specs claiming one name silently overwrite each other (this happened:
`gemma3-12b.json` said `Gemma3-4B-NPU2`).
`specs/open-engine/tests/test_spec_model_names.py` now refuses that, and refuses
a name that disagrees with the spec's own family and size.
