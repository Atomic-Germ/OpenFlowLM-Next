# Shared toolchain environment for open NPU kernel builds.
# Used by both the package derivation (open-kernels.nix) and the dev shell.
{
  lib,
  stdenv,
  fetchurl,
  python312,
  python312Packages,
  pkgs ? null,
  xrt ? pkgs.xrt,
  xrt-plugin-amdxdna ? pkgs.xrt-plugin-amdxdna,
  git,
  makeWrapper,
  unzip,
  zlib,
  patchelf,
}:

let
  pyVersion = "3.12";
  pySite = "lib/python${pyVersion}/site-packages";

  # XRT needs the amdxdna plugin next to its lib/ to enumerate NPU devices.
  # nix-amd-ai ships xrt and xrt-plugin-amdxdna separately; combine them the
  # same way the nix-amd-ai NixOS module does.
  xrtPrefix = "${xrt}/opt/xilinx/xrt";
  xrtCombined = stdenv.mkDerivation {
    pname = "xrt-combined";
    version = xrt.version;
    phases = [ "installPhase" ];
    installPhase = ''
      mkdir -p $out
      cp -rs ${xrtPrefix}/* $out/
      chmod -R u+w $out/lib
      ln -sf ${xrt-plugin-amdxdna}/opt/xilinx/xrt/lib/libxrt_driver_xdna* $out/lib/
    '';
  };

  mlirAieWheel = fetchurl {
    url = "https://github.com/Xilinx/mlir-aie/releases/download/v1.4.3/mlir_aie-1.4.3-cp312-cp312-manylinux_2_35_x86_64.whl";
    sha256 = "sha256-TYwjpbYXHdEWLHjFYPCkjK1llpC1vlMNUWZrf/9ScoE=";
  };

  llvmAieWheel = fetchurl {
    url = "https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026092301%2B02be6fd8-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    sha256 = "sha256-bi0uvoa/Z12aGcIhzUMCMtMUw5EEiUTeP8QVB5JANK4=";
  };

  pyDeps = with python312Packages; [
    aiofiles
    cloudpickle
    mdurl
    ml-dtypes
    numpy
    pygments
    rich
  ];

  # Runtime / native tools needed by both the package build and the dev shell.
  nativeTools = [ git makeWrapper python312 unzip patchelf ];
  runtimeLibs = [ stdenv.cc.cc.lib zlib xrt ];

  ironvenv = stdenv.mkDerivation {
    pname = "openflowlm-ironvenv";
    version = "1.4.3-20260923";

    nativeBuildInputs = [ python312 unzip ];
    src = ./.;
    dontUnpack = true;

    installPhase = ''
      ${python312}/bin/python -m venv $out
      export SITE="$out/${pySite}"

      ${unzip}/bin/unzip -q -d "$SITE" ${mlirAieWheel}
      ${unzip}/bin/unzip -q -d "$SITE" ${llvmAieWheel}

      # Link nixpkgs Python deps into the venv so imports resolve without
      # re-installing them or relying on .pth files.
      ${lib.concatStringsSep "\n" (map (pkg: "ln -s ${pkg}/${pySite}/* \"$SITE/\"") pyDeps)}

      # The wheel drops the aie package under mlir_aie/python; make it
      # importable without relying on aie.pth.
      ln -s "$SITE/mlir_aie/python/aie" "$SITE/aie"

      mkdir -p $out/nix-support
      echo "OpenFlowLM IRON toolchain venv (Python 3.12)" > $out/nix-support/hydra-build-products
    '';

    meta.platforms = [ "x86_64-linux" ];
  };
in
{
  inherit pyVersion pySite pyDeps nativeTools runtimeLibs ironvenv xrtCombined;

  env = {
    # Standard environment variables needed by export scripts at run time.
    # Use the combined XRT so the amdxdna plugin is discoverable next to lib/.
    XILINX_XRT = "${xrtCombined}";
    # Relative to the in-repo materialized venv; callers prepend $PWD/ironvenv/.
    PEANO_INSTALL_DIR = "${pySite}/llvm-aie";
    PATH = "${pySite}/llvm-aie/bin:${pySite}/mlir_aie/bin";
    PYTHONPATH = "${pySite}";
    LD_LIBRARY_PATH = lib.makeLibraryPath ([ xrtCombined ] ++ runtimeLibs);
  };
}
