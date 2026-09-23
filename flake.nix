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

      flake = {
        overlays.default = final: prev: {
          oflm = inputs.self.packages.${prev.system}.oflm;
          openflowlm-open-kernels = inputs.self.packages.${prev.system}.openflowlm-open-kernels;
          openflowlm-open-kernels-with-bert = inputs.self.packages.${prev.system}.openflowlm-open-kernels-with-bert;
        };

        nixosModules = {
          default = { config, lib, pkgs, ... }: {
            nixpkgs.overlays = [ inputs.self.overlays.default ];
            imports = [ ./nix/nixos-module.nix ];
          };
          openflowlm = { config, lib, pkgs, ... }: {
            nixpkgs.overlays = [ inputs.self.overlays.default ];
            imports = [ ./nix/nixos-module.nix ];
          };
        };
      };

      perSystem = { config, self', inputs', system, ... }: let
        pkgs = import inputs.nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ inputs.nix-amd-ai.overlays.default ];
        };
      in {
        packages = {
          oflm = pkgs.callPackage ./nix/package.nix {
            source = inputs.self;
            openflowlm-open-kernels = config.packages.openflowlm-open-kernels;
          };
          oflm-with-bert = pkgs.callPackage ./nix/package.nix {
            source = inputs.self;
            openflowlm-open-kernels = config.packages.openflowlm-open-kernels-with-bert;
          };
          openflowlm-open-kernels = pkgs.callPackage ./nix/open-kernels.nix { srcRoot = inputs.self; };
          openflowlm-open-kernels-with-bert = pkgs.callPackage ./nix/open-kernels.nix { srcRoot = inputs.self; skipBert = false; };
          default = config.packages.oflm;
        };

        apps.default = {
          type = "app";
          program = "${config.packages.oflm}/bin/oflm";
        };

        devShells = {
          default = config.devShells.oflm;

          oflm = let
            oflmPkg = pkgs.callPackage ./nix/package.nix {
              source = inputs.self;
              openflowlm-open-kernels = config.packages.openflowlm-open-kernels;
            };
          in pkgs.mkShell {
            name = "oflm-dev";
            nativeBuildInputs = with pkgs; [
              cmake
              ninja
              pkg-config
              patchelf
              cargo
              rustc
            ];
            buildInputs = oflmPkg.buildInputs;
            shellHook = ''
              export XILINX_XRT="${pkgs.xrt}/opt/xilinx/xrt"
              export PKG_CONFIG_PATH="${pkgs.xrt}/lib/pkgconfig''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

            '';
          };

          open-kernels = pkgs.callPackage ./nix/open-kernels-shell.nix {};
        };
      };
    };
}
