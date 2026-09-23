{
  description = "OpenFlowLM — open NPU kernels for Ryzen AI";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    nix-amd-ai = {
      url = "github:noamsto/nix-amd-ai";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      perSystem = { config, self', inputs', system, ... }: let
        pkgs = import inputs.nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ inputs.nix-amd-ai.overlays.default ];
        };
      in {
        packages = {
          oflm = pkgs.callPackage ./package.nix {};
          openflowlm-open-kernels = pkgs.callPackage ./open-kernels.nix {};
          openflowlm-open-kernels-with-bert = pkgs.callPackage ./open-kernels.nix { skipBert = false; };
          default = config.packages.oflm;
        };

        apps.default = {
          type = "app";
          program = "${config.packages.oflm}/bin/oflm";
        };

        devShells = {
          default = config.devShells.oflm;

          oflm = pkgs.mkShell {
            name = "oflm-dev";
            nativeBuildInputs = with pkgs; [
              cmake
              ninja
              pkg-config
              patchelf
              cargo
              rustc
            ];
            buildInputs = (pkgs.callPackage ./package.nix {}).buildInputs;
            shellHook = ''
              export XILINX_XRT="${pkgs.xrt}/opt/xilinx/xrt"
              export PKG_CONFIG_PATH="${pkgs.xrt}/lib/pkgconfig''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
            '';
          };

          open-kernels = pkgs.callPackage ./open-kernels-shell.nix {};
        };
      };
    };
}
