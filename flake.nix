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
        packages.oflm = pkgs.callPackage ./package.nix {};
        packages.default = config.packages.oflm;

        apps.default = {
          type = "app";
          program = "${config.packages.oflm}/bin/oflm";
        };
      };
    };
}
