# Nix packaging

`nix build github:Atomic-Germ/OpenFlowLM-Next#openflowlm` gives you
`result/bin/oflm`, plus `oflm-test` and `q4nx-build`.

## What it is

A derivation that **packages the release tarball** the release workflow
publishes, rather than compiling the engine inside Nix. See
[nix/default.nix](default.nix) for why: the AMD NPU runtime (XRT) is not in
nixpkgs, so a from-source build needs it vendored as a fixed-output derivation
with a hash that has to be refreshed by hand. The binary distribution already
binds its own ABI, and its RUNPATH is `$ORIGIN/../lib64`, so it drops into the
store unmodified.

## The pin

`nix/pin.json` names the version and the sha256 of the tarball. A flake cannot
take an argument, so the version is data:

```json
{ "version": "1.2.3", "sha256": "sha256-..." }
```

Each GitHub release carries its own `openflowlm-<version>-nix.tar.gz`, which is
this directory with `pin.json` written for that version. Use that for a specific
release:

```bash
nix build "https://github.com/Atomic-Germ/OpenFlowLM-Next/releases/download/v1.2.3/openflowlm-1.2.3-nix.tar.gz#openflowlm"
```

The repository's `nix/pin.json` tracks the **newest published** release, and is
refreshed in a follow-up commit after a release (the release workflow cannot do
it: that would be a commit after the tag). It is a placeholder until then, and
`nix build` on the repository says so in one sentence rather than failing with
a hash mismatch.

To use the repository flake for a specific older version, copy that release's
`pin.json` over this one (it is the only file that differs):

```bash
tar xzf openflowlm-1.0.0-nix.tar.gz   # contains openflowlm-1.0.0-nix/flake.nix + pin.json
cp openflowlm-1.0.0-nix/pin.json nix/pin.json
```

## In a flake

```nix
{
  inputs.openflowlm.url =
    "github:Atomic-Germ/OpenFlowLM-Next/releases/download/v1.2.3/openflowlm-1.2.3-nix.tar.gz";
  # ...
  environment.systemPackages = [ inputs.openflowlm.packages.x86_64-linux.openflowlm ];
}
```

or as an overlay:

```nix
nixpkgs.overlays = [ inputs.openflowlm.overlays.default ];
# then openflowlm, in environment.systemPackages
```

## What it does not do

* **No source build.** See above. If XRT is ever vendored for Nix, `default.nix`
  becomes a `callPackage` over a `cmakeReleaseHook` and the pin goes away.
* **No system integration.** `/etc/profile.d` and `/usr/bin/oflm` are in the
  RPM/DEB/TGZ because a package manager has to put them there; a Nix package
  does not, and `environment.systemPackages` is that mechanism.
* **No CPU-only fallback.** The NPU is AMD-only and so is XRT.

## Checking it

`nix flake check` is the same build on a different system attribute, and the
derivation's own `checkPhase` runs `oflm --version` and `oflm list` from the
store, which is the thing that catches a store move that broke the relative
`RUNPATH`. The release workflow runs it on every tag, so a flake that does not
build never ships.
