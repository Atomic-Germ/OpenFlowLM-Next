# Nix workflows for OpenFlowLM-Next

This repository is a flake.  All workflows below assume you are in the
repository root and have Nix with flakes enabled.

## TL;DR

```bash
# Imperative / on-the-fly
nix develop .#open-kernels
python utilities/export-kernels.py          # dense + BERT sets
nix run .#oflm -- --help                    # run the engine

# Declarative on NixOS
{
  programs.openflowlm = {
    enable = true;
    enableNPU = true;   # pulls in nix-amd-ai XRT + amdxdna plugin
  };
}
```

---

## 1. Building kernels imperatively

The kernel toolchain (mlir-aie, Peano, XRT+amdxdna plugin) is exposed
through the `open-kernels` dev shell.  This lets you build xclbins outside
the Nix sandbox, which is required for the BERT embedding sets because they
need `/dev/accel*` access at build time.

```bash
nix develop .#open-kernels

# Build everything (dense open_kernels + open_npue BERT sets)
python utilities/export-kernels.py

# Build only the BERT embedding sets
python utilities/export-kernels.py --bert-only

# Build dense specs, skipping known-failing ones
python utilities/export-kernels.py --skip-bert \
  --skip-specs qwen25-3b,minicpm5-2b,phi4-mini-4b --force
```

Built xclbins land in:
- `src/xclbins/<Model-Name-NPU2>/open_kernels/` for dense specs
- `src/xclbins/BERT-*-*/gemm_rtp/` for BERT sets

The shell materializes a writable copy of the IRON venv (`ironvenv/`) in the
checkout and points `PATH`/`PYTHONPATH` at it, so the export scripts need no
further setup.  The same shell is available without flakes via
`nix-shell nix/shell.nix`.

Note that the toolchain the Nix builds and shells use is pinned in
`nix/open-kernels-env.nix` (mlir-aie 1.4.3 + the matching Peano nightly, from
the upstream manylinux wheels) and is deliberately independent of
`ironvenv-requirements.txt`, which the non-Nix flow installs with pip.

---

## 2. Running the engine

### Default engine (dense kernels only)

```bash
nix run .#oflm -- --help
```

This builds and runs the engine with the default `openflowlm-open-kernels`
package.  The kernels are linked into the engine package's `share/oflm/xclbins`,
so no environment variable is needed to find them.  Because BERT needs NPU
access at build time, the default kernel package skips BERT so it can be built
in a sandboxed Nix build.

### Engine with BERT kernels

On a machine with an AMD XDNA2 NPU and the nix-amd-ai XRT+amdxdna plugin:

```bash
nix run .#oflm-with-bert -- --help
```

This bundles the engine with `openflowlm-open-kernels-with-bert`, which
enables the BERT embedding sets.

### Dev shell for engine development

```bash
nix develop .#oflm
```

This exposes the C++/CMake build inputs, with XRT combined with the amdxdna
plugin so an NPU host can actually run the engine.  The shell also sets
`OFLM_XCLBIN_PATH` to the default kernel package so engine builds/tests can
find xclbins without further configuration (a build tree has no
`share/oflm` next to the binary).

---

## 3. Declarative NixOS module

Import the flake in your NixOS configuration and enable the module:

```nix
{
  inputs.openflowlm.url = "github:Atomic-Germ/OpenFlowLM-Next";
  inputs.openflowlm.inputs.nixpkgs.follows = "nixpkgs";

  outputs = { self, nixpkgs, openflowlm, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        openflowlm.nixosModules.default
        {
          programs.openflowlm = {
            enable = true;
            package = openflowlm.packages.x86_64-linux.oflm;
            kernelsPackage = openflowlm.packages.x86_64-linux.openflowlm-open-kernels-with-bert;
          };
        }
      ];
    };
  };
}
```

`nixosModules.default` adds the flake's overlay and imports both
`nix-amd-ai.nixosModules.default` and this repo's module.  The nix-amd-ai
module loads the `amdxdna` kernel module, sets up `/dev/accel*` udev rules,
configures PAM memlock limits for the `video` and `render` groups, and wires
`XILINX_XRT` / `XRT_PATH` to the XRT + amdxdna plugin combination.  This is
what makes `xrt-smi` and `pyxrt` see the NPU.  Our module sets
`hardware.amd-npu.enable` to a default of `true`.

Hosts that already import nix-amd-ai or set `hardware.amd-npu.enable` are not
affected, because `lib.mkDefault` only supplies a default.

For build/CI hosts that do not have an NPU, disable the runtime wiring:

```nix
programs.openflowlm = {
  enable = true;
  enableNPU = false;
};
```

The engine and kernel packages are still installed; only the amdxdna module,
udev rules, and XRT plugin setup are skipped.

The kernel xclbins are merged from `kernelsPackage` and any
`extraXclbinPackages` into the engine package's own
`share/oflm/xclbins`, which is the first place the engine looks
(`<exe_dir>/../share/oflm`) before any environment variable is set. So on a
system build they live under:

```
/run/current-system/sw/share/oflm/xclbins
```

with the individual sets symlinked from the kernel packages in the store.
Models added per-user with `oflm add` land in `~/.config/oflm/xclbins`, which
the engine searches as well, so the two coexist without configuration.

### Adding BERT embedding sets

The `openflowlm-open-kernels-with-bert` package requires `__noChroot` because
the BERT build opens `/dev/accel*` at build time, so it cannot be built
inside a sandboxed `nixos-rebuild`.  On sandboxed NixOS builders use the
default `kernelsPackage` (dense kernels only) and add BERT xclbins from an
out-of-sandbox build:

```bash
# On the target NPU host
nix develop .#open-kernels
python utilities/export-kernels.py --bert-only
```

The engine will find the BERT sets automatically when run from the git
checkout because it searches `./xclbins`.  To make them available system-wide,
copy them into the engine's user-level search path:

```bash
mkdir -p ~/.config/oflm
rm -rf ~/.config/oflm/xclbins
ln -s /path/to/repo/src/xclbins ~/.config/oflm/xclbins
```

`~/.config/oflm/xclbins` is searched by `find_xclbin_path()` and merged with
the system-wide dense kernels installed by the module.

---

## 4. Flake packages and shells

| Flake output                               | Purpose                                            |
|--------------------------------------------|----------------------------------------------------|
| `.#oflm`                                   | Engine package (dense kernels)                       |
| `.#oflm-with-bert`                         | Engine package including BERT embedding sets       |
| `.#openflowlm-open-kernels`                | Dense open kernel xclbins only                     |
| `.#openflowlm-open-kernels-with-bert`      | Dense + BERT xclbins (needs NPU at build time)     |
| `.#oflm` dev shell                         | C++/CMake engine development                       |
| `.#open-kernels` dev shell                   | mlir-aie / IRON kernel toolchain                   |

---

### The bundled Python utilities need their own dependencies

`oflm-test`, `q4nx-build` and `oflm-add` ship in the package, and their
interpreters are patched to a Nix-provided Python, but their *third-party*
dependencies are not vendored — they are pip requirements of the two Python
projects, not C++ build inputs.  Out of the box:

```bash
nix run .#oflm -- list            # works
$out/bin/oflm-test --help         # ModuleNotFoundError: openai
$out/bin/q4nx-build --help        # ModuleNotFoundError: gguf
```

Give them an environment with what they import when you actually need them.
The wrappers append (not prepend) the packaged Python to `PATH`, so an
environment of your own wins:

```bash
nix shell nixpkgs#python3 -c python3 -m venv ~/.venvs/oflm-util
~/.venvs/oflm-util/bin/pip install openai                                                  # oflm-test
~/.venvs/oflm-util/bin/pip install gguf numpy torch einops safetensors huggingface-hub   # q4nx-build

PATH=~/.venvs/oflm-util/bin:$PATH oflm-test --help
```

`modelscope` is deliberately *not* in that list. It is only the fallback that
points at AMD's re-uploads of models (which may or may not run under
OpenFlowLM at all), and nixpkgs marks it insecure for CVE-2026-84202 — unsafe
YAML deserialization while loading a model config. Install it from PyPI
yourself if you want that fallback; the Hugging Face path needs nothing extra.

---

## 5. Known limitations

- Every dense spec in `open_kernels/recipes/specs/` builds with the toolchain
  pinned in `nix/open-kernels-env.nix`.  An earlier revision skipped
  `qwen25-3b`, `minicpm5-2b` and `phi4-mini-4b`; those recipe problems have
  since been fixed upstream.  If one regresses, the derivation's `skipSpecs`
  argument takes a comma-separated spec list (empty by default).
- `openflowlm-open-kernels-with-bert` requires `__noChroot` because the BERT
  build opens `/dev/accel*` at build time.  It will not build in a sandboxed
  `nix build` unless sandboxing is disabled; use `nix develop` and build
  imperatively, or enable it on a NixOS host with the NPU module.
