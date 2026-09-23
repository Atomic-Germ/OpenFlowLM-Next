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

---

## 2. Running the engine

### Default engine (dense kernels only)

```bash
nix run .#oflm -- --help
```

This builds and runs the engine with the default `openflowlm-open-kernels`
package.  Because BERT needs NPU access at build time, the default kernel
package skips BERT so it can be built in a sandboxed Nix build.

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

This exposes the C++/CMake build inputs.  The shell also sets
`OFLM_XCLBIN_PATH` to the default kernel package so engine builds/tests can
find xclbins without further configuration.

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

The module imports `nix-amd-ai.nixosModules.default` unconditionally.  It
sets `hardware.amd-npu.enable` to a default of `true`, which loads the
`amdxdna` kernel module, sets up `/dev/accel*` udev rules, configures PAM
memlock limits for the `video` and `render` groups, and wires `XILINX_XRT` /
`XRT_PATH` to the XRT + amdxdna plugin combination.  This is what makes
`xrt-smi` and `pyxrt` see the NPU.

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

The kernel xclbins are installed onto the system and appear under:

```
/run/current-system/sw/share/oflm/xclbins
```

The `oflm` binary is wrapped with `--set-default OFLM_XCLBIN_PATH` pointing
at the chosen kernel package, so it finds the kernels automatically.

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

## 5. Known limitations

- The dense specs `qwen25-3b`, `minicpm5-2b`, and `phi4-mini-4b` currently fail
  with the upstream mlir-aie 1.4.3 / Peano 20260923 toolchain.  They are
  skipped by default in the package build until the recipe is fixed.
- `openflowlm-open-kernels-with-bert` requires `__noChroot` because the BERT
  build opens `/dev/accel*` at build time.  It will not build in a sandboxed
  `nix build` unless sandboxing is disabled; use `nix develop` and build
  imperatively, or enable it on a NixOS host with the NPU module.
