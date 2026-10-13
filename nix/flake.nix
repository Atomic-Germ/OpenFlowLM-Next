{
  description = "OpenFlowLM - LLM inference with dense GEMM offloaded to the AMD NPU2";

  # nixpkgs-unstable rather than a release channel: this derivation only calls
  # stdenvNoCC/makeWrapper/patchelf/python3, all of which are stable API, so
  # there is nothing here a channel pin would protect, and unstable is a ref
  # that always exists.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      # The pin is a data file, not an argument, because a flake cannot take
      # one. `nix build .#openflowlm` has to mean a specific published version,
      # and the release workflow rewrites this file for the version it built.
      pin = builtins.fromJSON (builtins.readFile ./pin.json);

      # All-zero placeholder: a fake hash would fail at fetch time with a wall
      # of hex. Failing at eval time with one sentence is the difference between
      # "this flake is not pinned to a release" and "something is wrong with
      # nix's crypto".
      valid = builtins.match "sha256-[A-Za-z0-9+/]+={0,2}" pin.sha256 != null
              && pin.sha256 != "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";

      mk = pkgs: pkgs.callPackage ./default.nix {
        inherit (pin) version sha256;
      };

      systems = [ "x86_64-linux" "aarch64-linux" ];
    in
    {
      packages = nixpkgs.lib.genAttrs systems
        (system: {
          openflowlm = if valid then mk nixpkgs.legacyPackages.${system} else
            throw ''
              nix/pin.json does not point at a published release.
              It is written by the release workflow into the
              openflowlm-<version>-nix.tar.gz asset of every GitHub release; copy
              that file over nix/pin.json (or pass --override-input), or set
              version and sha256 by hand.
            '';
        });

      apps = nixpkgs.lib.genAttrs systems
        (system: {
          openflowlm = {
            type = "app";
            program = "${self.packages.${system}.openflowlm}/bin/oflm";
          };
        });

      overlays.default = final: _prev: {
        openflowlm = final.callPackage ./default.nix {
          inherit (pin) version sha256;
        };
      };

      # Building the package IS the check (nix/default.nix runs oflm --version
      # and oflm list in its checkPhase), so `nix flake check` is the same
      # build on a different system attribute.
      checks = nixpkgs.lib.genAttrs systems
        (system: { openflowlm = self.packages.${system}.openflowlm; });
    };
}
