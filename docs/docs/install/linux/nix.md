---
title: "Nix & NixOS"
description: "Install and run OpenFlowLM on Linux with Nix flakes, Home Manager, or the declarative NixOS module."
icon: "layers"
weight: 30
aliases: ["/docs/install/nix/"]
---

Install and run OpenFlowLM (`x86_64-linux`) via the `.#oflm` Nix flake, Home Manager, or the declarative `programs.openflowlm` NixOS module, with user models and settings stored in `~/.config/oflm/`.

## Quick Start

<div class="oflm-two-col">
  <div class="oflm-card">

### Run Directly via Flake

```bash
nix run \
  github:Atomic-Germ/OpenFlowLM-Next#oflm \
  -- --help

# Or from a local checkout:
nix run .#oflm -- list
```

  </div>
  <div class="oflm-card">

### Minimal NixOS Configuration

```nix
{
  programs.openflowlm = {
    enable = true;
    enableNPU = true;
  };
}
```

  </div>
</div>

## Declarative NixOS Module

Import `openflowlm.nixosModules.default` in your system `flake.nix` to configure both the AMD NPU hardware stack and OpenFlowLM in one step:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    openflowlm = {
      url = "github:Atomic-Germ/OpenFlowLM-Next";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, openflowlm, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        openflowlm.nixosModules.default
        {
          programs.openflowlm = {
            enable = true;
            package = openflowlm.packages.x86_64-linux.oflm;
            kernelsPackage = openflowlm.packages.x86_64-linux.openflowlm-open-kernels;
          };
        }
      ];
    };
  };
}
```

Enabling `programs.openflowlm` configures three components:

1. **AMD NPU driver & permissions (`hardware.amd-npu.enable`):** Loads the `amdxdna` kernel module (via [`nix-amd-ai`](https://github.com/noamsto/nix-amd-ai)), installs `/dev/accel*` udev rules, raises PAM `memlock` limits for `video`/`render` users, and configures the XRT userspace runtime.
2. **Bundled NPU acceleration:** Links `kernelsPackage` and any `extraXclbinPackages` into OpenFlowLM's search path with no extra environment variables.
3. **Writable user storage:** Models installed with `oflm add` and settings saved with `oflm config` live under `~/.config/oflm/`, separate from the read-only `/nix/store`.

| `programs.openflowlm` Option | Type | Default | Purpose |
|---|---|---|---|
| `enable` | `bool` | `false` | Install `oflm` in `environment.systemPackages` |
| `package` | `package` | `pkgs.oflm` | OpenFlowLM CLI and server package |
| `kernelsPackage` | `package` | `pkgs.openflowlm-open-kernels` | Bundled NPU acceleration package |
| `extraXclbinPackages` | `listOf package` | `[]` | Additional NPU kernel packages to merge (such as BERT encoders) |
| `enableNPU` | `bool` | `true` | Enable `hardware.amd-npu.enable` (`false` on headless CI machines) |

## Home Manager, Overlay & Prebuilt Release Flake

On standalone Nix or Home Manager setups (where your host OS provides the `amdxdna` kernel driver; see [Packages](/docs/install/linux/packages/)), apply `openflowlm.overlays.default` or add the flake package to `home.packages`:

```nix
{ inputs, pkgs, ... }: {
  home.packages = [
    inputs.openflowlm.packages.${pkgs.system}.oflm
  ];
}
```

To skip local compilation, [GitHub Releases](https://github.com/Atomic-Germ/OpenFlowLM-Next/releases) publish a prebuilt Nix tarball flake with bundled XRT libraries:

```bash
nix build "https://github.com/Atomic-Germ/OpenFlowLM-Next/releases/download/v<version>/openflowlm-<version>-nix.tar.gz#openflowlm"
./result/bin/oflm validate
```

### Flake Outputs (`x86_64-linux`)

| Output | Type | Description |
|---|---|---|
| `.#oflm` | Package | OpenFlowLM CLI, server, and open-source LLM/MoE/diffusion NPU kernels |
| `.#openflowlm-open-kernels` | Package | Source-built open NPU kernel bundle (`share/oflm/xclbins`) |
| `.#oflm-with-bert` | Package | Engine + BERT NPU embedding kernels (build with `--option sandbox false` on an NPU host) |
| `.#openflowlm-open-kernels-with-bert` | Package | Open NPU kernel bundle including BERT embedding encoders |
| `nixosModules.default` | NixOS Module | Declarative `programs.openflowlm` + `hardware.amd-npu` setup |
| `overlays.default` | Overlay | Adds `oflm` and open kernel packages to `pkgs` |
| `devShells.x86_64-linux.oflm` / `.open-kernels` | Dev Shells | Contributor shells for engine C++ builds and NPU kernel compilation |

## Model & Modality Availability on NixOS

The Nix flake compiles **only OpenFlowLM's open-source NPU kernels from source**. During packaging, `rm -rf src/xclbins` in `nix/package.nix` and `nix/open-kernels.nix` strips inherited upstream prebuilt `.xclbin` files that are being phased out ({{< b k="wip" issue="244" t="Issue #244" >}}).

- **Works out of the box on `.#oflm`:**
  - All open-kernel GGUF models installed via `oflm add` (Qwen3 / Qwen3.5 / Qwen3.6-MoE including CPU-ViT vision, Gemma 3 `4B`/`12B` text, Llama 3.1/3.2, Granite 4 `3B`, Phi-4-mini `4B`, LFM2 / LFM2.5, DeepSeek-R1 `8B`, and Nanbeige `3B`).
  - Base `.q4nx` shortcuts backed by open kernels (`qwen3.5:*`, `qwen3:*`, `llama3.*`, `gemma3:4b` text, and `gemma4-it:12b` text).
  - `flux2-klein:4b` image generation and editing (`oflm image` and `/v1/images/*`).
- **Requires `.#oflm-with-bert`:**
  - The 6 BERT-family NPU embedding models (`bge-*`, `all-minilm:l6-v2`, `nomic-embed-text:v1.5`, and `gte-multilingual:base`) compile their NPU artifacts against a live `/dev/accel/accel*` device outside the Nix sandbox:

    ```bash
    nix build .#oflm-with-bert --option sandbox false
    ```

- **Currently unavailable or degraded on NixOS ({{< b k="wip" issue="244" t="Issue #244" >}}):**
  - **Closed prebuilt `.xclbin` models/modalities:** `whisper-v3:turbo` ASR (until `export-kernels.py` builds open Whisper kernels under {{< b k="wip" issue="240" t="Issue #240" >}}), `gemma4-it:e2b` and `gemma4-it:e4b` ({{< b k="wip" issue="236" t="Issue #236" >}}), `gpt-oss:20b` ({{< b k="wip" issue="91" t="Issue #91" >}}), and closed Gemma vision/audio ({{< b k="wip" issue="73" t="Issue #73" >}}).
  - **CPU fallback on EmbeddingGemma:** `embed-gemma:300m` runs on CPU on NixOS because its prebuilt `npu_matmul_f32` `.xclbin` files are stripped ({{< b k="wip" issue="244" t="Issue #244" >}}).
  - **Omitted derivative `.q4nx` shortcut symlinks:** `qwen3-it:4b`, `qwen3-tk:4b`, `deepseek-r1:8b`, `deepseek-r1-0528:8b`, and `lfm2.5-*` lack kernel directory symlinks in `nix/package.nix` ({{< b k="wip" issue="244" t="Issue #244" >}}). Install their Hugging Face GGUF repositories directly with `oflm add` instead (for example, `oflm add unsloth/Qwen3-4B-Instruct-2507-GGUF`).

## Validate and Add a Model

After rebuilding your NixOS system (or building the flake), verify your NPU and install a GGUF model:

```bash
oflm validate
oflm add unsloth/Qwen3.5-4B-GGUF
oflm run qwen3.5:4b
```

> [!NOTE]
> `oflm add` uses only the Python standard library and works out of the box on NixOS. Contributors working on engine code or building custom NPU kernels with `nix develop` can see [Build System](/docs/contributing/build/).

See [Packages](/docs/install/linux/packages/), [Docker](/docs/install/linux/docker/), and [CLI Usage](/docs/instructions/cli/) for next steps.
