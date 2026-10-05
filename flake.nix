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
          # nix-amd-ai owns the NPU kernel module, udev rules, PAM memlock
          # limits, and the XRT + amdxdna plugin wiring; programs.openflowlm
          # below sets hardware.amd-npu.enable (the option it defines). It is
          # imported here rather than from nix/nixos-module.nix because a NixOS
          # module has no `self` argument -- resolving one from _module.args
          # recurses infinitely -- while `inputs` is in scope here.
          default = { config, lib, pkgs, ... }: {
            nixpkgs.overlays = [ inputs.self.overlays.default ];
            imports = [ inputs.nix-amd-ai.nixosModules.default ./nix/nixos-module.nix ];
          };
          openflowlm = { config, lib, pkgs, ... }: {
            nixpkgs.overlays = [ inputs.self.overlays.default ];
            imports = [ inputs.nix-amd-ai.nixosModules.default ./nix/nixos-module.nix ];
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
            # XRT's plugin loader resolves libxrt_driver_xdna next to
            # libxrt_core, so the dev shell needs the amdxdna plugin combined
            # with XRT -- plain `xrt` cannot enumerate the NPU.
            xrt-combined = pkgs.runCommand "xrt-combined" {} ''
              mkdir -p $out
              cp -rs ${pkgs.xrt}/opt/xilinx/xrt/* $out/
              chmod -R u+w $out/lib
              ln -sf ${pkgs.xrt-plugin-amdxdna}/opt/xilinx/xrt/lib/libxrt_driver_xdna* $out/lib/
            '';
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
              export XILINX_XRT="${xrt-combined}"
              export PKG_CONFIG_PATH="${xrt-combined}/lib/pkgconfig''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
              # Build-tree runs find no share/oflm next to the binary, so point
              # the engine at the kernel package while developing.
              export OFLM_XCLBIN_PATH="${config.packages.openflowlm-open-kernels}/share/oflm"
            '';
          };

          open-kernels = pkgs.callPackage ./nix/open-kernels-shell.nix {};
        };
      };
    };
}
