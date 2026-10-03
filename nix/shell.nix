# Standalone `nix-shell` entry point for the open-kernels toolchain, so the
# shell also works without flakes:
#
#   nix-shell nix/shell.nix
#
# The flake's `.#open-kernels` dev shell is the same shell; it is built here so
# there is one definition of the toolchain environment.
{ pkgs ? import <nixpkgs> { config.allowUnfree = true; }
, nix-amd-ai ? (builtins.getFlake (toString ./..)).inputs.nix-amd-ai
}:

let
  pkgs' = import pkgs.path {
    inherit (pkgs) system config;
    overlays = [ nix-amd-ai.overlays.default ];
  };
in

pkgs'.callPackage ./open-kernels-shell.nix { }
